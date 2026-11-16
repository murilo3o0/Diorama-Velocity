package dev.diorama.render.gpu;

import static dev.diorama.render.gpu.GpuTileBaker.FACE_BYTES;
import static dev.diorama.render.gpu.GpuTileBaker.CASTER_BYTES;
import static dev.diorama.render.gpu.GpuTileBaker.SCENE;
import static dev.diorama.render.gpu.GpuTileBaker.SHADOW_SIZE;
import static dev.diorama.render.gpu.GpuTileBaker.SKY_HEIGHT;
import static dev.diorama.render.gpu.GpuTileBaker.SKY_WIDTH;
import static dev.diorama.render.gpu.GpuTileBaker.VOXEL_HEIGHT;
import static dev.diorama.render.gpu.GpuTileBaker.VOXEL_WIDTH;
import static org.lwjgl.opengl.GL33C.*;

import dev.diorama.Diorama;
import dev.diorama.Projection;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.jspecify.annotations.Nullable;
import org.lwjgl.opengl.GL;
import org.lwjgl.opengl.GLCapabilities;
import org.lwjgl.system.MemoryStack;

final class GlBakeBackend implements BakeBackend {
	private static final String[] ATTRIBUTES = {"Corner", "Cell", "Color", "Occlusion", "FaceLight", "SkyIndex", "Box", "Haze"};
	private static final int UBO_BINDING = 30;
	private static final int UNITS = 6;
	private static final Pattern LOCATION = Pattern.compile("layout\\(location = \\w+\\) ?");
	private static final Pattern OUTPUT = Pattern.compile("layout\\(location = (\\d+)\\) out \\w+ (\\w+);");

	private @Nullable Boolean usable;
	private int skyProgram;
	private int shadowProgram;
	private int opaqueProgram;
	private int tintedProgram;
	private int waterProgram;
	private int compositeProgram;
	private int shadowTexture;
	private int sceneColor;
	private int sceneAux;
	private int sceneDepth;
	private int tint;
	private int waterColor;
	private int waterDepth;
	private int sky;
	private int voxels;
	private int output;
	private int outputInfo;
	private int skyFbo;
	private int shadowFbo;
	private int sceneFbo;
	private int tintFbo;
	private int waterFbo;
	private int outputFbo;
	private int faceVao;
	private int casterVao;
	private int emptyVao;
	private int cornerVbo;
	private int indexBuffer;
	private int faceVbo;
	private int casterVbo;
	private int ubo;
	private int faceVboCapacity;
	private int casterVboCapacity;
	private boolean timestamps;
	private @Nullable String device;
	private static boolean coreDivisor;
	private static boolean samplerObjects;

	private final ArrayDeque<Readback> readbacks = new ArrayDeque<>();
	private final ArrayDeque<Integer> colorPboPool = new ArrayDeque<>();
	private final ArrayDeque<Integer> infoPboPool = new ArrayDeque<>();

	private record Readback(GpuTileBaker.Job job, int colors, int info, long fence, int startQuery, int endQuery, long recordNanos) {
	}

	private int acquireColorPbo(int bytes) {
		Integer pooled = this.colorPboPool.pollFirst();
		int pbo = pooled != null ? pooled : glGenBuffers();
		glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
		if (pooled == null) {
			glBufferData(GL_PIXEL_PACK_BUFFER, bytes, GL_STREAM_READ);
		}
		return pbo;
	}

	private int acquireInfoPbo(int bytes) {
		Integer pooled = this.infoPboPool.pollFirst();
		int pbo = pooled != null ? pooled : glGenBuffers();
		glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
		if (pooled == null) {
			glBufferData(GL_PIXEL_PACK_BUFFER, bytes, GL_STREAM_READ);
		}
		return pbo;
	}

	private void releasePbos(int colors, int info) {
		if (colors != 0) {
			if (this.colorPboPool.size() < 16) {
				this.colorPboPool.addLast(colors);
			} else {
				glDeleteBuffers(colors);
			}
		}
		if (info != 0) {
			if (this.infoPboPool.size() < 16) {
				this.infoPboPool.addLast(info);
			} else {
				glDeleteBuffers(info);
			}
		}
	}

	@Override
	public boolean draw(GpuTileBaker.Job job) {
		if (!this.ready()) {
			return false;
		}
		long start = System.nanoTime();
		Saved saved = Saved.take();
		int startQuery = 0;
		int colors = 0;
		int info = 0;
		int endQuery = 0;
		boolean queued = false;
		try {
			startQuery = this.timestamps ? glGenQueries() : 0;
			if (startQuery != 0) {
				glQueryCounter(startQuery, GL_TIMESTAMP);
			}
			GlState.noCullNoScissor();
			GlState.colorMaskAll();
			GlState.blend(false);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
			glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

			glBindBuffer(GL_UNIFORM_BUFFER, this.ubo);
			glBufferData(GL_UNIFORM_BUFFER, GpuTileBaker.UBO_SIZE, GL_STREAM_DRAW);
			glBufferSubData(GL_UNIFORM_BUFFER, 0, job.uniforms);
			glBindBufferBase(GL_UNIFORM_BUFFER, UBO_BINDING, this.ubo);

			int faceBytes = job.faces.remaining();
			glBindBuffer(GL_ARRAY_BUFFER, this.faceVbo);
			if (faceBytes > this.faceVboCapacity) {
				this.faceVboCapacity = Math.max(faceBytes, this.faceVboCapacity * 2);
			}
			glBufferData(GL_ARRAY_BUFFER, this.faceVboCapacity, GL_STREAM_DRAW);
			if (faceBytes > 0) {
				glBufferSubData(GL_ARRAY_BUFFER, 0, job.faces);
			}

			int casterBytes = job.casters.remaining();
			glBindBuffer(GL_ARRAY_BUFFER, this.casterVbo);
			if (casterBytes > this.casterVboCapacity) {
				this.casterVboCapacity = Math.max(casterBytes, this.casterVboCapacity * 2);
			}
			glBufferData(GL_ARRAY_BUFFER, this.casterVboCapacity, GL_STREAM_DRAW);
			if (casterBytes > 0) {
				glBufferSubData(GL_ARRAY_BUFFER, 0, job.casters);
			}
			GlState.forgetBuffers();

			int skyFaces = job.opaqueCount + job.waterCount;
			if (job.voxels != null && skyFaces > 0) {
				this.bindTextures(this.voxels);
				Saved.plainPixelStore();
				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, VOXEL_WIDTH, job.voxelRows, GL_RED_INTEGER, GL_UNSIGNED_INT, job.voxels);
				this.target(this.skyFbo, SKY_WIDTH, SKY_HEIGHT);
				GlState.depthTest(false);
				GlState.useProgram(this.skyProgram);
				GlState.bindVertexArray(this.faceVao);
				if (job.opaqueCount > 0) {
					this.pointFaces(0);
					glDrawArraysInstanced(GL_POINTS, 0, 4, job.opaqueCount);
				}
				if (job.waterCount > 0) {
					this.pointFaces(job.opaqueCount + job.tintedCount);
					glDrawArraysInstanced(GL_POINTS, 0, 4, job.waterCount);
				}
			}

			this.target(this.shadowFbo, SHADOW_SIZE, SHADOW_SIZE);
			this.clearDepth();
			if (job.casterCount > 0) {
				this.depth(true);
				GlState.useProgram(this.shadowProgram);
				GlState.bindVertexArray(this.casterVao);
				glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0, job.casterCount);
			}

			this.target(this.sceneFbo, SCENE, SCENE);
			this.clearColor(0, 0.01F, 0.01F, 0.015F, 1);
			this.clearColor(1, 0, 0, 0, 0);
			this.clearDepth();
			this.bindTextures(this.shadowTexture, this.sky, this.voxels);
			GlState.bindVertexArray(this.faceVao);
			if (job.opaqueCount > 0) {
				this.depth(true);
				GlState.useProgram(this.opaqueProgram);
				this.pointFaces(0);
				glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0, job.opaqueCount);
			}

			this.target(this.tintFbo, SCENE, SCENE);
			this.clearColor(0, 0, 0, 0, 1);
			if (job.tintedCount > 0) {
				this.depth(false);
				GlState.blend(true);
				GlState.blendFuncSeparate(GL_DST_ALPHA, GL_ONE, GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);
				GlState.useProgram(this.tintedProgram);
				this.pointFaces(job.opaqueCount);
				glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0, job.tintedCount);
				GlState.blend(false);
			}

			this.target(this.waterFbo, SCENE, SCENE);
			this.clearColor(0, 0, 0, 0, 0);
			this.clearDepth();
			if (job.waterCount > 0) {
				this.depth(true);
				GlState.useProgram(this.waterProgram);
				this.pointFaces(job.opaqueCount + job.tintedCount);
				glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0, job.waterCount);
			}

			int size = Projection.TILE_PIXELS;
			this.target(this.outputFbo, size, size);
			GlState.depthTest(false);
			GlState.useProgram(this.compositeProgram);
			this.bindTextures(this.sceneColor, this.sceneDepth, this.sceneAux, this.tint, this.waterColor, this.waterDepth);
			GlState.bindVertexArray(this.emptyVao);
			glDrawArrays(GL_TRIANGLES, 0, 3);

			GlState.bindFramebuffer(GL_READ_FRAMEBUFFER, this.outputFbo);
			colors = this.acquireColorPbo(size * size * 4);
			Saved.plainPixelStore();
			glReadBuffer(GL_COLOR_ATTACHMENT0);
			glReadPixels(0, 0, size, size, GL_RGBA, GL_UNSIGNED_BYTE, 0L);

			if (job.mesh.overlays != null || !job.mesh.lit) {
				info = this.acquireInfoPbo(size * size * 2);
				glPixelStorei(GL_PACK_ALIGNMENT, 1);
				glReadBuffer(GL_COLOR_ATTACHMENT1);
				glReadPixels(0, 0, size, size, GL_RG, GL_UNSIGNED_BYTE, 0L);
				glReadBuffer(GL_COLOR_ATTACHMENT0);
			}
			if (startQuery != 0) {
				endQuery = glGenQueries();
				glQueryCounter(endQuery, GL_TIMESTAMP);
			}
			long fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
			long recordNanos = System.nanoTime() - start;
			this.readbacks.add(new Readback(job, colors, info, fence, startQuery, endQuery, recordNanos));
			queued = true;
			job.pending();
			GpuTileBaker.recorded(recordNanos);
		} finally {
			if (!queued) {
				this.releasePbos(colors, info);
				for (int query : new int[] {startQuery, endQuery}) {
					if (query != 0) {
						glDeleteQueries(query);
					}
				}
			}
			saved.restore();
			job.free();
		}
		return true;
	}

	@Override
	public @Nullable String device() {
		if (this.device == null) {
			this.device = glGetString(GL_RENDERER) + " (" + glGetString(GL_VENDOR) + ", OpenGL " + glGetString(GL_VERSION) + ")";
		}
		return this.device;
	}

	@Override
	public void poll() {
		if (this.readbacks.isEmpty()) {
			return;
		}
		int pack = glGetInteger(GL_PIXEL_PACK_BUFFER_BINDING);
		int processed = 0;
		try {
			while (!this.readbacks.isEmpty() && processed < 2) {
				Readback r = this.readbacks.peek();
				int status = glClientWaitSync(r.fence, 0, 0);
				if (status == GL_TIMEOUT_EXPIRED) {
					return;
				}
				this.readbacks.poll();
				glDeleteSync(r.fence);
				if (status == GL_WAIT_FAILED) {
					this.free(r);
					r.job.fail();
					continue;
				}
				try {
					int size = Projection.TILE_PIXELS;
					int[] pixels = new int[size * size];
					glBindBuffer(GL_PIXEL_PACK_BUFFER, r.colors);
					ByteBuffer mapped = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, (long) size * size * 4, GL_MAP_READ_BIT);
					if (mapped == null) {
						throw new IllegalStateException("couldn't map the tile's pixels");
					}
					mapped.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(pixels);
					glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
					byte[] info = null;
					byte[] light = null;
					if (r.info != 0) {
						glBindBuffer(GL_PIXEL_PACK_BUFFER, r.info);
						ByteBuffer bits = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, (long) size * size * 2, GL_MAP_READ_BIT);
						if (bits == null) {
							throw new IllegalStateException("couldn't map the tile's overlay bits");
						}
						byte[] both = new byte[size * size * 2];
						bits.get(both);
						glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
						info = r.job.mesh.overlays != null ? GpuTileBaker.channel(both, 0) : null;
						light = !r.job.mesh.lit ? GpuTileBaker.channel(both, 1) : null;
					}
					long gpu = -1;
					if (r.startQuery != 0 && glGetQueryObjecti(r.endQuery, GL_QUERY_RESULT_AVAILABLE) != 0) {
						gpu = glGetQueryObjecti64(r.endQuery, GL_QUERY_RESULT) - glGetQueryObjecti64(r.startQuery, GL_QUERY_RESULT);
					}
					r.job.done(pixels, info, light, r.recordNanos, gpu);
					processed++;
				} catch (Throwable t) {
					Diorama.LOGGER.error("Reading back GPU tile {} failed", r.job.mesh.key, t);
					r.job.fail();
				} finally {
					this.free(r);
				}
			}
		} finally {
			glBindBuffer(GL_PIXEL_PACK_BUFFER, pack);
		}
	}

	private void free(Readback r) {
		this.releasePbos(r.colors, r.info);
		if (r.startQuery != 0) {
			glDeleteQueries(r.startQuery);
			glDeleteQueries(r.endQuery);
		}
	}

	private void pointFaces(int first) {
		long offset = (long) first * FACE_BYTES;
		glBindBuffer(GL_ARRAY_BUFFER, this.faceVbo);
		glVertexAttribIPointer(1, 4, GL_SHORT, FACE_BYTES, offset);
		glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, true, FACE_BYTES, offset + 8);
		glVertexAttribPointer(3, 4, GL_UNSIGNED_BYTE, true, FACE_BYTES, offset + 12);
		glVertexAttribPointer(4, 4, GL_UNSIGNED_BYTE, true, FACE_BYTES, offset + 16);
		glVertexAttribIPointer(5, 1, GL_INT, FACE_BYTES, offset + 20);
		glVertexAttribIPointer(6, 2, GL_INT, FACE_BYTES, offset + 24);
		glVertexAttribIPointer(7, 4, GL_INT, FACE_BYTES, offset + 32);
	}

	private void target(int fbo, int width, int height) {
		GlState.bindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
		GlState.viewport(width, height);
	}

	private void depth(boolean write) {
		GlState.depthTest(true);
		GlState.depthFunc(GL_GEQUAL);
		GlState.depthMask(write);
	}

	private void clearDepth() {
		GlState.depthMask(true);
		try (MemoryStack stack = MemoryStack.stackPush()) {
			glClearBufferfv(GL_DEPTH, 0, stack.floats(0));
		}
	}

	private void clearColor(int attachment, float r, float g, float b, float a) {
		try (MemoryStack stack = MemoryStack.stackPush()) {
			glClearBufferfv(GL_COLOR, attachment, stack.floats(r, g, b, a));
		}
	}

	private void bindTextures(int... textures) {
		for (int i = 0; i < textures.length; i++) {
			GlState.activeTexture(i);
			GlState.bindTexture(textures[i]);
			noSampler(i);
		}
	}

	private boolean ready() {
		if (this.usable == null) {
			this.usable = false;
			try {
				GLCapabilities caps = GL.getCapabilities();
				coreDivisor = caps.glVertexAttribDivisor != 0;
				if (!coreDivisor && !caps.GL_ARB_instanced_arrays) {
					Diorama.LOGGER.warn("GPU baking needs instanced vertex attributes (OpenGL 3.3); tracing tiles instead");
					return false;
				}
				samplerObjects = caps.glBindSampler != 0;
				this.timestamps = caps.glQueryCounter != 0 && caps.glGetQueryObjecti64v != 0;
				Diorama.LOGGER.info("GPU baking on {}", this.device());
				int error;
				for (int i = 0; i < 16 && (error = glGetError()) != GL_NO_ERROR; i++) {
					Diorama.LOGGER.debug("OpenGL error {} was flagged before GPU baking set up; not counting it", error);
				}
				Saved saved = Saved.take();
				try {
					this.create(caps);
				} finally {
					saved.restore();
				}
				this.usable = true;
			} catch (Throwable t) {
				Diorama.LOGGER.error("Couldn't set up GPU baking; tracing tiles instead", t);
			}
		}
		return this.usable;
	}

	private void create(GLCapabilities caps) throws IOException {
		boolean zeroToOne = (caps.OpenGL45 || caps.GL_ARB_clip_control) && glGetInteger(0x935D) == 0x935F;
		String depth = zeroToOne ? "DIORAMA_DEPTH_ZERO_TO_ONE" : null;
		String vsh = shader("core/bake.vsh");
		String fsh = shader("core/bake.fsh");
		this.skyProgram = program(vsh, fsh, Map.of("Voxels", 0), "SKY_PASS", depth);
		this.shadowProgram = program(vsh, fsh, Map.of(), "SHADOW", depth);
		Map<String, Integer> faces = Map.of("ShadowMap", 0, "Sky", 1);
		this.opaqueProgram = program(vsh, fsh, Map.of("ShadowMap", 0, "Sky", 1, "Voxels", 2), null, depth);
		this.tintedProgram = program(vsh, fsh, faces, "TINTED", depth);
		this.waterProgram = program(vsh, fsh, Map.of("ShadowMap", 0, "Sky", 1, "Voxels", 2), "WATER", depth);
		Map<String, Integer> composite = new LinkedHashMap<>();
		String[] names = {"SceneColor", "SceneDepth", "SceneAux", "Tint", "WaterColor", "WaterDepth"};
		for (int i = 0; i < names.length; i++) {
			composite.put(names[i], i);
		}
		this.compositeProgram = program(shader("core/bake_composite.vsh"), shader("core/bake_composite.fsh"), composite, null, depth);

		int size = Projection.TILE_PIXELS;
		this.shadowTexture = texture(GL_DEPTH_COMPONENT32F, SHADOW_SIZE, SHADOW_SIZE, GL_DEPTH_COMPONENT, GL_FLOAT);
		this.sceneColor = texture(GL_RGBA16F, SCENE, SCENE, GL_RGBA, GL_HALF_FLOAT);
		this.sceneAux = texture(GL_RGBA16F, SCENE, SCENE, GL_RGBA, GL_HALF_FLOAT);
		this.sceneDepth = texture(GL_DEPTH_COMPONENT32F, SCENE, SCENE, GL_DEPTH_COMPONENT, GL_FLOAT);
		this.tint = texture(GL_RGBA16F, SCENE, SCENE, GL_RGBA, GL_HALF_FLOAT);
		this.waterColor = texture(GL_RGBA16F, SCENE, SCENE, GL_RGBA, GL_HALF_FLOAT);
		this.waterDepth = texture(GL_DEPTH_COMPONENT32F, SCENE, SCENE, GL_DEPTH_COMPONENT, GL_FLOAT);
		this.sky = texture(GL_R8, SKY_WIDTH, SKY_HEIGHT, GL_RED, GL_UNSIGNED_BYTE);
		this.voxels = texture(GL_R32UI, VOXEL_WIDTH, VOXEL_HEIGHT, GL_RED_INTEGER, GL_UNSIGNED_INT);
		this.output = texture(GL_RGBA8, size, size, GL_RGBA, GL_UNSIGNED_BYTE);
		this.outputInfo = texture(GL_RG8, size, size, GL_RG, GL_UNSIGNED_BYTE);

		this.skyFbo = framebuffer(0, this.sky);
		this.shadowFbo = framebuffer(this.shadowTexture);
		this.sceneFbo = framebuffer(this.sceneDepth, this.sceneColor, this.sceneAux);
		this.tintFbo = framebuffer(this.sceneDepth, this.tint);
		this.waterFbo = framebuffer(this.waterDepth, this.waterColor);
		this.outputFbo = framebuffer(0, this.output, this.outputInfo);

		this.ubo = glGenBuffers();
		this.faceVbo = glGenBuffers();
		this.casterVbo = glGenBuffers();
		this.faceVboCapacity = 256 * 1024;
		this.casterVboCapacity = 64 * 1024;
		this.cornerVbo = glGenBuffers();
		glBindBuffer(GL_ARRAY_BUFFER, this.cornerVbo);
		try (MemoryStack stack = MemoryStack.stackPush()) {
			glBufferData(GL_ARRAY_BUFFER, stack.bytes((byte) 0, (byte) 0, (byte) 0, (byte) 0, (byte) 255, (byte) 0, (byte) 0, (byte) 0,
				(byte) 255, (byte) 255, (byte) 0, (byte) 0, (byte) 0, (byte) 255, (byte) 0, (byte) 0), GL_STATIC_DRAW);
			this.indexBuffer = glGenBuffers();
			this.faceVao = glGenVertexArrays();
			GlState.bindVertexArray(this.faceVao);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this.indexBuffer);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, stack.shorts((short) 0, (short) 1, (short) 2, (short) 2, (short) 3, (short) 0), GL_STATIC_DRAW);
		}
		this.corners();
		for (int a = 1; a < ATTRIBUTES.length; a++) {
			glEnableVertexAttribArray(a);
			divisor(a, 1);
		}
		this.pointFaces(0);
		this.casterVao = glGenVertexArrays();
		GlState.bindVertexArray(this.casterVao);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this.indexBuffer);
		this.corners();
		glBindBuffer(GL_ARRAY_BUFFER, this.casterVbo);
		glEnableVertexAttribArray(1);
		divisor(1, 1);
		glVertexAttribIPointer(1, 4, GL_SHORT, CASTER_BYTES, 0);
		glEnableVertexAttribArray(6);
		divisor(6, 1);
		glVertexAttribIPointer(6, 2, GL_INT, CASTER_BYTES, 8);
		this.emptyVao = glGenVertexArrays();
		GlState.forgetBuffers();
		int error = glGetError();
		if (error != GL_NO_ERROR) {
			throw new IllegalStateException("OpenGL error " + error + " setting up GPU baking");
		}
	}

	private static void divisor(int attribute, int divisor) {
		if (coreDivisor) {
			glVertexAttribDivisor(attribute, divisor);
		} else {
			org.lwjgl.opengl.ARBInstancedArrays.glVertexAttribDivisorARB(attribute, divisor);
		}
	}

	private static void noSampler(int unit) {
		if (samplerObjects) {
			glBindSampler(unit, 0);
		}
	}

	private void corners() {
		glBindBuffer(GL_ARRAY_BUFFER, this.cornerVbo);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_UNSIGNED_BYTE, true, 4, 0);
		divisor(0, 0);
	}

	private static int texture(int internalFormat, int width, int height, int format, int type) {
		int texture = glGenTextures();
		GlState.activeTexture(0);
		GlState.bindTexture(texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
		glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, (ByteBuffer) null);
		return texture;
	}

	private static int framebuffer(int depth, int... colors) {
		int fbo = glGenFramebuffers();
		GlState.bindFramebuffer(GL_FRAMEBUFFER, fbo);
		if (depth != 0) {
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);
		}
		int[] buffers = new int[colors.length];
		for (int i = 0; i < colors.length; i++) {
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colors[i], 0);
			buffers[i] = GL_COLOR_ATTACHMENT0 + i;
		}
		if (colors.length == 0) {
			glDrawBuffer(GL_NONE);
			glReadBuffer(GL_NONE);
		} else {
			glDrawBuffers(buffers);
		}
		int status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			throw new IllegalStateException("GPU bake framebuffer incomplete: " + status);
		}
		return fbo;
	}

	private static String shader(String name) throws IOException {
		String text = BLOCK_COMMENT.matcher(read("/assets/diorama/shaders/" + name)).replaceAll("");
		String include = java.util.regex.Matcher.quoteReplacement(BLOCK_COMMENT.matcher(read("/assets/diorama/shaders/include/bake.glsl")).replaceAll(""));
		return INCLUDE.matcher(text).replaceAll(include);
	}

	private static final Pattern BLOCK_COMMENT = Pattern.compile("/\\*.*?\\*/", Pattern.DOTALL);
	private static final Pattern INCLUDE = Pattern.compile("(?m)^#(include|moj_import) <diorama:bake\\.glsl>$");

	private static String read(String path) throws IOException {
		try (InputStream in = GlBakeBackend.class.getResourceAsStream(path)) {
			if (in == null) {
				throw new IOException("missing " + path);
			}
			return new String(in.readAllBytes(), StandardCharsets.UTF_8);
		}
	}

	private static int program(String vsh, String fsh, Map<String, Integer> samplers, @Nullable String define, @Nullable String depth) {
		Map<String, Integer> outputs = new LinkedHashMap<>();
		Matcher m = OUTPUT.matcher(fsh);
		while (m.find()) {
			outputs.put(m.group(2), Integer.parseInt(m.group(1)));
		}
		String defines = (define == null ? "" : "#define " + define + "\n") + (depth == null ? "" : "#define " + depth + "\n");
		int vs = compile(GL_VERTEX_SHADER, prepare(vsh, defines));
		int fs = compile(GL_FRAGMENT_SHADER, prepare(fsh, defines));
		int program = glCreateProgram();
		glAttachShader(program, vs);
		glAttachShader(program, fs);
		for (int a = 0; a < ATTRIBUTES.length; a++) {
			glBindAttribLocation(program, a, ATTRIBUTES[a]);
		}
		for (Map.Entry<String, Integer> output : outputs.entrySet()) {
			glBindFragDataLocation(program, output.getValue(), output.getKey());
		}
		glLinkProgram(program);
		glDetachShader(program, vs);
		glDetachShader(program, fs);
		glDeleteShader(vs);
		glDeleteShader(fs);
		if (glGetProgrami(program, GL_LINK_STATUS) == GL_FALSE) {
			String log = glGetProgramInfoLog(program);
			glDeleteProgram(program);
			throw new IllegalStateException("GPU bake shader didn't link (" + define + "): " + log);
		}
		int block = glGetUniformBlockIndex(program, "DioramaBake");
		if (block != GL_INVALID_INDEX) {
			glUniformBlockBinding(program, block, UBO_BINDING);
		}
		GlState.useProgram(program);
		for (Map.Entry<String, Integer> sampler : samplers.entrySet()) {
			int location = glGetUniformLocation(program, sampler.getKey());
			if (location >= 0) {
				glUniform1i(location, sampler.getValue());
			}
		}
		return program;
	}

	private static String prepare(String source, String defines) {
		String text = LOCATION.matcher(source).replaceAll("")
			.replace("#version 330", "#version 150")
			.replace("#extension GL_ARB_separate_shader_objects : require", "")
			.replace("gl_VertexIndex", "gl_VertexID");
		int line = text.indexOf('\n');
		return text.substring(0, line + 1) + defines + text.substring(line + 1);
	}

	private static int compile(int type, String source) {
		int shader = glCreateShader(type);
		glShaderSource(shader, source);
		glCompileShader(shader);
		if (glGetShaderi(shader, GL_COMPILE_STATUS) == GL_FALSE) {
			String log = glGetShaderInfoLog(shader);
			glDeleteShader(shader);
			throw new IllegalStateException("GPU bake shader didn't compile: " + log);
		}
		return shader;
	}

	private record Saved(
		int program, int vao, int arrayBuffer, int uniformBuffer, int uboBinding, int pack, int unpack, int drawFbo, int readFbo,
		int[] viewport, int activeTexture, int[] textures, int[] samplers, int[] pixelStore,
		boolean cull, boolean scissor, boolean depthTest, int depthFunc, boolean depthMask, boolean blend0, boolean blend1, int[] blendFunc, boolean[] colorMask
	) {
		private static final int[] PIXEL_STORE = {
			GL_PACK_ALIGNMENT, GL_PACK_ROW_LENGTH, GL_PACK_SKIP_PIXELS, GL_PACK_SKIP_ROWS,
			GL_UNPACK_ALIGNMENT, GL_UNPACK_ROW_LENGTH, GL_UNPACK_SKIP_PIXELS, GL_UNPACK_SKIP_ROWS
		};

		static Saved take() {
			int[] viewport = new int[4];
			glGetIntegerv(GL_VIEWPORT, viewport);
			int active = glGetInteger(GL_ACTIVE_TEXTURE) - GL_TEXTURE0;
			int[] textures = new int[UNITS];
			int[] samplers = new int[UNITS];
			for (int u = 0; u < UNITS; u++) {
				GlState.activeTexture(u);
				textures[u] = glGetInteger(GL_TEXTURE_BINDING_2D);
				samplers[u] = samplerObjects ? glGetInteger(GL_SAMPLER_BINDING) : 0;
			}
			GlState.activeTexture(active);
			int[] pixelStore = new int[PIXEL_STORE.length];
			for (int i = 0; i < PIXEL_STORE.length; i++) {
				pixelStore[i] = glGetInteger(PIXEL_STORE[i]);
			}
			boolean[] colorMask = new boolean[4];
			try (MemoryStack stack = MemoryStack.stackPush()) {
				ByteBuffer mask = stack.malloc(4);
				glGetBooleanv(GL_COLOR_WRITEMASK, mask);
				for (int i = 0; i < 4; i++) {
					colorMask[i] = mask.get(i) != 0;
				}
			}
			int[] blendFunc = {
				glGetInteger(GL_BLEND_SRC_RGB), glGetInteger(GL_BLEND_DST_RGB), glGetInteger(GL_BLEND_SRC_ALPHA), glGetInteger(GL_BLEND_DST_ALPHA)
			};
			return new Saved(
				glGetInteger(GL_CURRENT_PROGRAM), glGetInteger(GL_VERTEX_ARRAY_BINDING), glGetInteger(GL_ARRAY_BUFFER_BINDING),
				glGetInteger(GL_UNIFORM_BUFFER_BINDING), glGetIntegeri(GL_UNIFORM_BUFFER_BINDING, UBO_BINDING),
				glGetInteger(GL_PIXEL_PACK_BUFFER_BINDING), glGetInteger(GL_PIXEL_UNPACK_BUFFER_BINDING),
				glGetInteger(GL_DRAW_FRAMEBUFFER_BINDING), glGetInteger(GL_READ_FRAMEBUFFER_BINDING), viewport, active, textures, samplers, pixelStore,
				glIsEnabled(GL_CULL_FACE), glIsEnabled(GL_SCISSOR_TEST), glIsEnabled(GL_DEPTH_TEST), glGetInteger(GL_DEPTH_FUNC),
				glGetBoolean(GL_DEPTH_WRITEMASK), glIsEnabledi(GL_BLEND, 0), glIsEnabledi(GL_BLEND, 1), blendFunc, colorMask
			);
		}

		static void plainPixelStore() {
			for (int i = 0; i < PIXEL_STORE.length; i++) {
				int name = PIXEL_STORE[i];
				glPixelStorei(name, name == GL_PACK_ALIGNMENT || name == GL_UNPACK_ALIGNMENT ? 4 : 0);
			}
		}

		void restore() {
			for (int u = 0; u < UNITS; u++) {
				GlState.activeTexture(u);
				GlState.bindTexture(this.textures[u]);
				if (samplerObjects) {
					glBindSampler(u, this.samplers[u]);
				}
			}
			GlState.activeTexture(this.activeTexture);
			GlState.useProgram(this.program);
			GlState.bindVertexArray(this.vao);
			glBindBuffer(GL_ARRAY_BUFFER, this.arrayBuffer);
			glBindBufferBase(GL_UNIFORM_BUFFER, UBO_BINDING, this.uboBinding);
			glBindBuffer(GL_UNIFORM_BUFFER, this.uniformBuffer);
			glBindBuffer(GL_PIXEL_PACK_BUFFER, this.pack);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, this.unpack);
			GlState.bindFramebuffer(GL_DRAW_FRAMEBUFFER, this.drawFbo);
			GlState.bindFramebuffer(GL_READ_FRAMEBUFFER, this.readFbo);
			GlState.viewportAt(this.viewport[0], this.viewport[1], this.viewport[2], this.viewport[3]);
			for (int i = 0; i < PIXEL_STORE.length; i++) {
				glPixelStorei(PIXEL_STORE[i], this.pixelStore[i]);
			}
			GlState.cull(this.cull);
			GlState.scissor(this.scissor);
			GlState.depthTest(this.depthTest);
			GlState.depthFunc(this.depthFunc);
			GlState.depthMask(this.depthMask);
			GlState.blendFuncSeparate(this.blendFunc[0], this.blendFunc[1], this.blendFunc[2], this.blendFunc[3]);
			GlState.restoreBlend(this.blend0, this.blend1);
			GlState.colorMask(this.colorMask[0], this.colorMask[1], this.colorMask[2], this.colorMask[3]);
			GlState.forgetBuffers();
		}
	}
}
