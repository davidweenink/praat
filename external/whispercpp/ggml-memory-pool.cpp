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

#include "ggml-memory-pool.h"
#include "ggml-impl.h"
#include "melder.h"

void GgmlMemoryPool :: add(void *ptr, size_t size = 0) {
	TRACE
	if (ptr == nullptr) {
		trace (U"Trying to add nullptr to allocations)");
		return;
	}
	if (allocations .count(ptr))
		trace (U"Something is very wrong: pointer already in allocations (possible double allocation or missing remove)");

	allocations [ptr] = size;
}

void GgmlMemoryPool :: remove(void *ptr) {
	if (ptr == nullptr)
		return;
	allocations .erase (ptr);
}

void GgmlMemoryPool :: clear() {
	if (allocations.empty())
		return;
	for (auto & allocation : allocations) {
		if (allocation .first == nullptr)
			continue;
		if (allocation .second)
			ggml_aligned_free (allocation .first, allocation .second);
		else
			free (allocation .first);
	}
	allocations.clear();
}
