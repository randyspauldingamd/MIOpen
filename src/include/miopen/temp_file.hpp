/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef GUARD_TEMP_FILE_HPP
#define GUARD_TEMP_FILE_HPP

#include <miopen/tmp_dir.hpp>

#include <string>

namespace miopen {

class MIOPEN_INTERNALS_EXPORT TempFile
{
public:
    TempFile(const std::string& path_infix);
    TempFile(TempFile&& other) noexcept = default;
    TempFile& operator=(TempFile&& other) noexcept = default;

    const std::string& GetPathInfix() const { return path_infix; }
    fs::path Path() const { return dir.path / "file"; }
    operator fs::path() const { return Path(); }

private:
    std::string path_infix;
    TmpDir dir;
};
} // namespace miopen

#endif
