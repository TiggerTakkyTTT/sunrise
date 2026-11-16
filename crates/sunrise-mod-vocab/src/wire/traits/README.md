# Wire call-table roles

Every trait that crosses the wasm boundary declares its role in one file, named
for the root-exported trait:

- `TraitName.import`: a trait the game implements and a mod calls;
- `TraitName.export`: a trait a mod implements (a callback, a Part, a handler);
- `TraitName.held`: an export that the game also hands a mod objects of, which
  the mod calls back through the `held-<interface>` imports.

The file's text is a short note on why the trait crosses; the filename is the
machine-consumed identity. Duplicate roles and invalid names fail generation.
The generator resolves the name to a real public API trait and builds both sides
of its call table. Adding, removing or renaming a role file reruns generation.

There is no other list. The call table follows file-name order, so the call ids
are the position in that order: adding a trait moves the ids after it, which is
why a mod is rebuilt against each game build. The handler traits that
`mod_api`'s capability keys name are added from the keys, as held exports.

This does not define a value layout. Public data types explicitly derive Wire.
Callback trait objects use the existing shared resource codec at their own
trait declaration, because traits cannot derive Wire.
