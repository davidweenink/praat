#ifndef _ggml_memory_pool_h_
#define _ggml_memory_pool_h_
/* ggml-memory-pool.cpp
 *
 * Copyright (C) 2026 Anastasia Shchupak
 *
 * This code is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 *
 * This code is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this work. If not, see <http://www.gnu.org/licenses/>.
 */

/*
* A memory tracking pool for GGML memory allocations within Praat.
 *
 * All memory allocations by GGML are added to this pool. If an allocation fails, instead of abort
 * (which causes Praat to crash), we will free all the memory registered in the pool. This will allow
 * Praat to continue running after a graceful end of whatever was using GGML: transcription or diarizarion.
 */

#include <unordered_map>

class GgmlMemoryPool {
private:
	std::unordered_map <void *, size_t> allocations;

public:
	void add(void *ptr, size_t size);   // registers a successful allocation
	void remove(void *ptr);   // removes allocation after freeing the memory normally; does nothing if *ptr was not registered
	void clear();   // free all registered allocations and clear the pool
};

/* End of file ggml-memory-pool.h */
#endif