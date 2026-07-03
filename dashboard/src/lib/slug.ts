import GithubSlugger from "github-slugger";

// A fresh slugger per call: GithubSlugger de-dupes repeated headings by
// appending -1/-2, which would drift from GitHub's own per-page numbering
// if we reused one instance across unrelated heading lists.
export function githubHeadingSlug(heading: string): string {
  return new GithubSlugger().slug(heading);
}
