# SD-Material

Substance Designer procedural material studies. Each folder is a self-contained
`.sbs` graph (with its `.resources` bitmap inputs where applicable). This is an
ongoing study collection — from early tiling exercises to fully art-directed,
production-oriented materials.

- **Software:** Substance 3D Designer
- **Format:** `.sbs` (editable graphs), `.sbsar` (published), `.tif` (bitmap inputs)
- **Not included:** Marmoset `.tbscene` render scenes, exported `.png/.tga`
  bakes, third-party reference libraries, and `.autosave` backups.

## Index

### Ground & Terrain
| Folder | Material | Graph |
| --- | --- | --- |
| `test1_方砖地形` | Square-tile ground / terrain | `test1.sbs`, `TileGround.sbs` |
| `test2_人字红砖` | Herringbone red brick | `test2.sbs` |
| `test19_dizhuan` | Floor tile | `test19_dizhuan.sbs` |
| `test33_地面练习` | Ground study | `BasePractice.sbs` |
| `test34_泥地` | Muddy road | `MudRoad.sbs`, `MudRoad2.sbs` |
| `test37_条纹泥` | Striped mud | `StripeMud.sbs` |
| `test38_AI路` | Road (AI-assisted workflow) | `AI_Road.sbs` |

### Stone & Rock
| Folder | Material | Graph |
| --- | --- | --- |
| `test3_大理石碎片` | Cracked marble | `test3.sbs` |
| `test4_鹅卵石群` | Cobblestone cluster | `Cobblestone.sbs` |
| `test5_雪地寒石` | Snow-covered cold stone | `SnowStone.sbs` |
| `test31_石头练习` | Rock study (6 variants) | `Stone1–6.sbs` |
| `test35_鹅卵石` | Cobblestone | `Cobblestone.sbs` |
| `test36_鹅卵石（干）` | Cobblestone (dry) | `ELS.sbs` |
| `test39_Houdini石头` | Rock (Houdini + SD pipeline) | `Houdini_Stone.sbs` |

### Fabric & Surface
| Folder | Material | Graph |
| --- | --- | --- |
| `test8_沙发皮质` | Sofa leather | `test8.sbs` |
| `test10_刺绣花纹` | Embroidery pattern | `test10.sbs` |
| `test11_刺绣细节` | Embroidery detail | `test11.sbs` |
| `test12_刺绣细节2` | Embroidery detail 2 | `test12.sbs` |
| `test13_刺绣细节3` | Embroidery detail 3 | `test13.sbs` |
| `test16_毛发` | Fur / hair surface | `毛发.sbs`, `毛发2.sbs` |

### Nature & Organic
| Folder | Material | Graph |
| --- | --- | --- |
| `test25_Bamboo` | Bamboo | `bamboo.sbs` |
| `test27_GrassGround` | Grass ground | `GrassGround.sbs` |
| `test32_树皮` | Tree bark | `Bark.sbs` |

### Architectural
| Folder | Material | Graph |
| --- | --- | --- |
| `test14_瓦窑纹理` | Kiln tile texture | `test14.sbs` |
| `test20_stair` | Stairs | `test20_stair.sbs` |
| `test28_roof` | Roof tiles | `Roof.sbs`, `Roof2.sbs` |
| `test30_墙皮` | Weathered wall plaster | `Wall.sbs` |

### Sci-fi & Hard Surface
| Folder | Material | Graph |
| --- | --- | --- |
| `test7_科幻概念` | Sci-fi concept | `test7.sbs` |
| `test21_it` | Hard-surface panel | `test21_it.sbs` |
| `test23_CircuitBoard` | Circuit board | `CircuitBoard.sbs` |
| `test26_scifi` | Sci-fi surface | `Sci-fi_01.sbs`, `CreasedTest.sbs` |

### Stylized & Misc
| Folder | Material | Graph |
| --- | --- | --- |
| `test6_冰面` | Ice surface | `test6.sbs` |
| `test15_破碎玻璃` | Broken glass | `broken_glass.sbs`, `broken_glass2.sbs` |
| `test17_扑克` | Playing cards | `test17_puke.sbs` |
| `test18_crystal` | Crystal | `test18_crystal.sbs` |
| `test22_hotpot` | Hotpot | `Hotpot.sbs` |
| `test24_Chess` | Chess set | `Chess.sbs` |
| `test29_风格化练习` | Stylized studies | `Stylized*.sbs` (4 graphs) |

### Special Techniques
| Folder | Material | Graph |
| --- | --- | --- |
| `SD_Hair` | Hair card workflow (HairTG plugins) | `HairCard.sbs` + `HairTG-*.sbsar` |

## Opening the source

Open any `.sbs` in Substance 3D Designer. Graphs that rely on external bitmap
inputs keep them in the matching `*.resources/` folder, so keep each project
folder intact when moving files.
