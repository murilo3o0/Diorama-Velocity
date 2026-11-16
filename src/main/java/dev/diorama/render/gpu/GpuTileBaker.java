// =========================================================================
// SUBSTITUA APENAS O MÉTODO drawQueued EM:
// src/main/java/dev/diorama/render/gpu/GpuTileBaker.java
// =========================================================================

	private static boolean drawQueued(long now) {
		boolean playing = SavedChunkImport.playing();
		if (lastFrame != 0) {
			double activeShare = playing ? Math.min(SHARE, 0.04) : SHARE;
			double maxSavings = playing ? 1.2 * cost() : 3.0 * cost();
			budget = Math.min(budget + activeShare * (now - lastFrame), maxSavings);
			double backgroundShare = playing ? Math.min(BACKGROUND_SHARE, 0.015) : SHARE;
			backgroundBudget = Math.min(backgroundBudget + backgroundShare * (now - lastFrame), cost());
		}
		lastFrame = now;
		// Durante o jogo ativo, nunca desenha mais de 1 tile por frame para zerar picos de frametime
		int maxDraws = playing ? 1 : 4;
		int draws = 0;
		boolean drew = false;
		while (budget >= cost() && draws < maxDraws) {
			Job job = QUEUE.poll();
			if (job == null) {
				break;
			}
			if (draw(job)) {
				budget -= cost();
				drew = true;
				draws++;
			}
		}
		if (!QUEUE.isEmpty() || draws >= maxDraws) {
			return drew;
		}
		while (backgroundBudget >= cost() && draws < maxDraws) {
			Job job = BACKGROUND.poll();
			if (job == null) {
				return drew;
			}
			if (draw(job)) {
				backgroundBudget -= cost();
				drew = true;
				draws++;
			}
		}
		return drew;
	}
