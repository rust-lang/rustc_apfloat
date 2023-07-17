# Detailed Licensing Information

This crate started as a port of LLVM's APFloat and (parts of) APInt code (revision [`f3598e8fca83ccfb11f58ec7957c229e349765e3`](https://github.com/llvm/llvm-project/commit/f3598e8fca83ccfb11f58ec7957c229e349765e3)).
At that time, LLVM was licensed under the [University of Illinois/NCSA Open Source License](https://spdx.org/licenses/NCSA.html).
LLVM has since worked to [relicense](https://foundation.llvm.org/docs/relicensing/) their project under the [Apache 2](https://spdx.org/licenses/Apache-2.0.html) with [LLVM Exception](https://spdx.org/licenses/LLVM-exception.html) license.

Reviewing the history of the APFloat/APInt code in LLVM, of the 795 commits which modify the code, only the following 7 have yet to be relicensed:

- [`cb4029110040c3655a66b5f423d328c749ba6a49`](https://github.com/llvm/llvm-project/commit/cb4029110040c3655a66b5f423d328c749ba6a49)
- [`f907b891da1641034f0603b0c6bc00b7aa4d1f4a`](https://github.com/llvm/llvm-project/commit/f907b891da1641034f0603b0c6bc00b7aa4d1f4a)
- [`bf19e0a5561593d3d25924693a20a9bbe7771a3f`](https://github.com/llvm/llvm-project/commit/bf19e0a5561593d3d25924693a20a9bbe7771a3f)
- [`49c758b769b9a787d415a9c2ce0e40fa0e482412`](https://github.com/llvm/llvm-project/commit/49c758b769b9a787d415a9c2ce0e40fa0e482412)
- [`4e69e29a72a1ffcbf755f13ed909b51cfbcafd60`](https://github.com/llvm/llvm-project/commit/4e69e29a72a1ffcbf755f13ed909b51cfbcafd60)
- [`8710d9831b1a78e3ecb1f49da24447ee27f73096`](https://github.com/llvm/llvm-project/commit/8710d9831b1a78e3ecb1f49da24447ee27f73096)
- [`1c419ff50d35c4cab442f5a1c8f5e82812921633`](https://github.com/llvm/llvm-project/commit/1c419ff50d35c4cab442f5a1c8f5e82812921633)

However, as the [LLVM blog mentions](https://blog.llvm.org/posts/2022-08-14-relicensing-update/):

> Some pieces of code are not covered by copyright law.
> For example, copyright law contains a concept called [“Threshold of originality”](https://en.wikipedia.org/wiki/Threshold_of_originality).
> It means that a work needs to be “sufficiently original” for it to be considered to be covered by copyright.
> There could be a lot of different interpretations into what it means for a code contribution to be sufficiently original for it to be covered by copyright.
> A threshold that is often used in open source projects that use [contributor license agreements (CLA)](https://en.wikipedia.org/wiki/Contributor_License_Agreement) is to assume that any contribution that’s 10 lines of code or less does not meet the threshold of originality and therefore copyright does not apply.
> In [their May 2022](https://discourse.llvm.org/t/board-meeting-minutes-may-2022/63628) board meeting, the LLVM Foundation decided to make the same assumption for the relicensing project: contributions of 10 lines of code or less are assumed to not be covered by copyright.
> Therefore, we don’t need relicensing agreements for those.
>
> Furthermore, there are a few commits that don’t seem to meet the “threshold-of-originality” even though they’re changing/adding more than 10 lines.
> We also consider those to not needing a relicensing agreement.
> One example is [this commit](https://github.com/llvm/llvm-project/commit/cd13ef01a21e), which only removes the full stop at the end of a few sentences.

With that in mind, a review of the commits in question shows:

### [`cb4029110040c3655a66b5f423d328c749ba6a49`](https://github.com/llvm/llvm-project/commit/cb4029110040c3655a66b5f423d328c749ba6a49)

This commit is the result of running a spellchecker on the code base.
While it is larger than 10 lines of code, it is very similar to the example commit mentioned above which mechanically removed the full stop from the end of various sentences in the codebase.
As such, it does not seem "sufficiently original" to be copyrightable.

### [`f907b891da1641034f0603b0c6bc00b7aa4d1f4a`](https://github.com/llvm/llvm-project/commit/f907b891da1641034f0603b0c6bc00b7aa4d1f4a)

This commit fixes hyphenation of words mostly in code comments or documentation.
Again, while it is larger than 10 lines of code, it is very similar to the example commit mentioned above and does not seem "sufficiently original" to be copyrightable.

### [`bf19e0a5561593d3d25924693a20a9bbe7771a3f`](https://github.com/llvm/llvm-project/commit/bf19e0a5561593d3d25924693a20a9bbe7771a3f)

This commit fixes a comparison to work properly with the MSVC compiler.
As the total diff is only 4 lines, it does not meet the established threshold of originality.

### [`49c758b769b9a787d415a9c2ce0e40fa0e482412`](https://github.com/llvm/llvm-project/commit/49c758b769b9a787d415a9c2ce0e40fa0e482412)

This commit changes some uses of raw pointers to use `unique_ptr`.
While the commit is larger than 10 lines of code, it is reverted in the next commit which landed the following day.
As such, the combination of this commit and the following commit are zero lines of code changed in the codebase.

### [`4e69e29a72a1ffcbf755f13ed909b51cfbcafd60`](https://github.com/llvm/llvm-project/commit/4e69e29a72a1ffcbf755f13ed909b51cfbcafd60)

This commit is a mechanical revert of the prior commit.

### [`8710d9831b1a78e3ecb1f49da24447ee27f73096`](https://github.com/llvm/llvm-project/commit/8710d9831b1a78e3ecb1f49da24447ee27f73096)

This commit changes a `struct` definition to be a `class`.
As the total diff is only 2 lines, it does not meet the established threshold of originality.

### [`1c419ff50d35c4cab442f5a1c8f5e82812921633`](https://github.com/llvm/llvm-project/commit/1c419ff50d35c4cab442f5a1c8f5e82812921633)

This commit fixes an assertion in the code which had not been correctly updated in a prior change.
As the total diff is only 6 lines of code (excluding changes to APInt unit tests which we did not port to Rust), it does not meet the established threshold of originality.

# Conclusion

The original LLVM code appears to be available as Apache 2 with LLVM Exception and so our port of this code is licensed as such.

A few additional patches (code cleanups and performance improvements) have been made on top of the initial port.
The authors of these patches have also agreed to allow their code to be used under the Apache 2 with LLVM Exception license.

Subsequent work on this crate has advanced the state of the port from the original commit.
Reviewing the set of upstream LLVM changes after revision `f3598e8fca83ccfb11f58ec7957c229e349765e3` and before the relicensing on 2019-01-19 reveals 41 changes all of which LLVM has relicensing agreements with their authors.
As such, these changes and all changes made to LLVM after the relicensing data are available under the Apache 2 with LLVM Exception license.

Therefore, the whole of this crate is Apache 2 with LLVM Exception licensed.
