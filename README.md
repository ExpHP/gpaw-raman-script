# gpaw-raman-script

Implements https://www.nature.com/articles/s41467-020-16529-6

(Note: since starting development on this, gpaw now also has its own implementation of this method currently in development, but it doesn't allow symmetry-reduced displacements)

## Guided example

https://gist.github.com/ExpHP/dce34c5008a0a1dffaf2bf2a1dfe7db9

## Notice

Note: currently depends on this specific patched form of gpaw:

* https://gitlab.com/ExpHP/gpaw/-/commits/myu-bugfix

(in the time since then, gpaw has implemented its own fix for the bug, but ASE also implemented a huge breaking change in force file formats so we cannot use that yet)
