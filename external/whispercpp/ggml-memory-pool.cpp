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

GgmlMemoryPool theGgmlMemoryPool;

void GgmlMemoryPool :: add(void *ptr, size_t size, bool aligned) {
	//TRACE
	if (! ptr) {
		trace (U"Trying to add nullptr to allocations)");
		return;
	}
	if (allocations.count (ptr))
		trace (U"Something is very wrong: pointer already in allocations (possible double allocation or missing remove)");

	allocations [ptr] = { size, aligned };
}

void GgmlMemoryPool :: remove(void *ptr) {
	//TRACE
	if (! ptr) {
		trace (U"Trying to remove nullptr from allocations)");
		return;
	}
	auto it = allocations.find (ptr);
	Melder_assert (it != allocations.end ());   // trying to remove an allocation which does not exist
	allocations .erase (ptr);
}

void GgmlMemoryPool :: remove(void *ptr, size_t size) {
	//TRACE
	if (! ptr) {
		trace (U"Trying to remove nullptr from allocations)");
		return;
	}

	auto it = allocations.find (ptr);
	Melder_assert (it != allocations.end ());   // trying to remove an allocation which does not exist
	Melder_assert (size == it->second.size);   // otherwise something weird: size does not match
	allocations .erase (ptr);
}

void GgmlMemoryPool :: clear() {
	//TRACE
	trace (U"Clearing allocations...");
	for (auto & allocation_pair : allocations) {
		if (! allocation_pair.first)
			continue;
		trace (U"Emergency freeing: ", allocation_pair.second.size, U" bytes at ", Melder_pointer (allocation_pair.first),
				allocation_pair.second.aligned ? U", aligned" : U"");
		if (allocation_pair.second.aligned)
			ggml_aligned_free (allocation_pair.first, allocation_pair.second.size);
		else
			ggml_raw_free (allocation_pair.first);
	}
	trace (U"Allocations cleared");
	allocations.clear();
}

size_t GgmlMemoryPool :: size() const {
	return allocations.size();
}
