# Changelog

All notable changes to this project will be documented in this file. 


<!-- this is a change log template -->
<!-- ## 9.4.0 (2021-12-31)
### Features
* add .cjs config file ([issue link: #717](https://github.com/conventional-changelog/standard-version/issues/717)) ([commit id: eceaedf](https://github.com/conventional-changelog/standard-version/commit/eceaedf8b3cdeb282ee06bfa9c65503f42404858))

### Bug Fixes

* description here. ([issue link: #534](https://github.com/conventional-changelog/standard-version/issues/534)) ([commit id: 2785023](https://github.com/conventional-changelog/standard-version/commit/2785023c91668e7300e6a22e55d31b6bd9dae59b)), closes [close issue: #533](https://github.com/conventional-changelog/standard-version/issues/533)

### ⚠ BREAKING CHANGES

* changes here.  -->

## 0.1.5 (2023-10-13)
Refactor the implementation of `StrImpl`. 

### Features
* Refactor `StrImpl` with `Box<str>` instead of flexiable array.
* Refactor `UserDataImpl` with intermidiate pointer instead of flexiable array.

### Fixes
* Extract some logic into function in `ManagedHeap`.
* rewrite `full_gc` method to fix unexpected collect for newborned object.

## 0.1.4 (2023-10-11)
Refactor the implementation of `CodeGen`. 

### Features
* Refactor `CodeGen` to support some optimization in `luac`.  
* Add more source location infomation to AST to generate correct line number during code generation.

## 0.1.2 (2023-09-08)
Add GC support, which implemented as a stop-the-world full collector.

### Features
* A usable GC.  
* Stablize `State::script_file(file)`.  


## 0.1.1 (2023-08-29)
First runnable version with a few testing.  

### Features
* An usable parser which has passed all tests.  
* Stablize `State::script(src)`.  

