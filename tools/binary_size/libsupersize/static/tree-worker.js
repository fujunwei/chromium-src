// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// @ts-check
'use strict';

/**
 * @fileoverview
 * Web worker code to parse JSON data from binary_size into data for the UI to
 * display.
 */

/**
 * @typedef {object} Meta
 * @prop {string[]} components
 * @prop {number} total
 * @prop {boolean} diff_mode
 */
/**
 * @typedef {object} SymbolEntry JSON object representing a single symbol.
 * @prop {string} n Name of the symbol.
 * @prop {number} b Byte size of the symbol, divided by num_aliases.
 * @prop {string} t Single character string to indicate the symbol type.
 * @prop {number} [u] Count value indicating how many symbols this entry
 * represents. Negative value when removed in a diff.
 */
/**
 * @typedef {object} FileEntry JSON object representing a single file and its
 * symbols.
 * @prop {string} p Path to the file (source_path).
 * @prop {number} c Index of the file's component in meta (component_index).
 * @prop {SymbolEntry[]} s - Symbols belonging to this node. Array of objects.
 */

importScripts('./shared.js');

const _PATH_SEP = '/';

/** @param {FileEntry} fileEntry */
function getSourcePath(fileEntry) {
  return fileEntry[_KEYS.SOURCE_PATH];
}

/**
 * Find the last index of either '/' or `sep` in the given path.
 * @param {string} path
 * @param {string} sep
 */
function lastIndexOf(path, sep) {
  if (sep === _PATH_SEP) {
    return path.lastIndexOf(_PATH_SEP);
  } else {
    return Math.max(path.lastIndexOf(sep), path.lastIndexOf(_PATH_SEP));
  }
}

/**
 * Return the dirname of the pathname 'path'. In a file path, this is the
 * full path of its folder.
 * @param {string} path Path to find dirname of.
 * @param {string} sep Path seperator, such as '/'.
 */
function dirname(path, sep) {
  return path.substring(0, lastIndexOf(path, sep));
}

/**
 * Compare two nodes for sorting. Used in sortTree.
 * @param {TreeNode} a
 * @param {TreeNode} b
 */
function _compareFunc(a, b) {
  return Math.abs(b.size) - Math.abs(a.size);
}

/**
 * Make a node with some default arguments
 * @param {Partial<TreeNode>} options
 * Values to use for the node. If a value is
 * omitted, a default will be used instead.
 * @returns {TreeNode}
 */
function createNode(options) {
  const {idPath, type, shortNameIndex, size = 0, childStats = {}} = options;
  return {
    children: [],
    parent: null,
    childStats,
    idPath,
    shortNameIndex,
    size,
    type,
  };
}

/**
 * Class used to build a tree from a list of symbol objects.
 * Add each file node using `addFileEntry()`, then call `build()` to finalize
 * the tree and return the root node. The in-progress tree can be obtained from
 * the `rootNode` property.
 */
class TreeBuilder {
  /**
   * @param {object} options
   * @param {(fileEntry: FileEntry) => string} options.getPath Called to get the
   * id path of a symbol's file entry.
   * @param {(symbolNode: TreeNode) => boolean} options.filterTest Called to see
   * if a symbol should be included. If a symbol fails the test, it will not be
   * attached to the tree.
   * @param {string} options.sep Path seperator used to find parent names.
   */
  constructor(options) {
    this._getPath = options.getPath;
    this._filterTest = options.filterTest;
    this._sep = options.sep || _PATH_SEP;

    this.rootNode = createNode({
      idPath: this._sep,
      shortNameIndex: 0,
      type: this._containerType(this._sep),
    });
    /** @type {Map<string, TreeNode>} Cache for directory nodes */
    this._parents = new Map();

    /**
     * Regex used to split the `idPath` when finding nodes. Equivalent to
     * one of: "/" or |sep|
     */
    this._splitter = new RegExp(`[/${this._sep}]`);
  }

  /**
   * Link a node to a new parent. Will go up the tree to update parent sizes to
   * include the new child.
   * @param {TreeNode} node Child node.
   * @param {TreeNode} directParent New parent node.
   */
  static _attachToParent(node, directParent) {
    // Link the nodes together
    directParent.children.push(node);
    node.parent = directParent;

    const additionalSize = node.size;
    const additionalStats = Object.entries(node.childStats);

    // Update the size and childStats of all ancestors
    while (node.parent != null) {
      const {parent} = node;
      const [containerType, lastBiggestType] = parent.type;
      let {size: lastBiggestSize = 0} =
        parent.childStats[lastBiggestType] || {};
      for (const [type, stat] of additionalStats) {
        const parentStat = parent.childStats[type] || {size: 0, count: 0};

        parentStat.size += stat.size;
        parentStat.count += stat.count;
        parent.childStats[type] = parentStat;

        const absSize = Math.abs(parentStat.size);
        if (absSize > lastBiggestSize) {
          parent.type = `${containerType}${type}`;
          lastBiggestSize = absSize;
        }
      }

      parent.size += additionalSize;
      node = parent;
    }
  }

  /**
   * Merges dex method symbols such as "Controller#get" and "Controller#set"
   * into containers, based on the class of the dex methods.
   * @param {TreeNode} node
   */
  static _joinDexMethodClasses(node) {
    const hasDexMethods = node.childStats[_DEX_METHOD_SYMBOL_TYPE] != null;
    if (!hasDexMethods || node.children == null) return node;

    if (node.type[0] === _CONTAINER_TYPES.FILE) {
      /** @type {Map<string, TreeNode>} */
      const javaClassContainers = new Map();
      /** @type {TreeNode[]} */
      const otherSymbols = [];

      // Place all dex methods into buckets
      for (const childNode of node.children) {
        // Java classes are denoted with a "#", such as "LogoView#onDraw"
        const splitIndex = childNode.idPath.lastIndexOf('#');

        const isDexMethodWithClass =
            childNode.type === _DEX_METHOD_SYMBOL_TYPE &&
            splitIndex > childNode.shortNameIndex;

        if (isDexMethodWithClass) {
          // Get the idPath of the class
          const classIdPath = childNode.idPath.slice(0, splitIndex);

          let classNode = javaClassContainers.get(classIdPath);
          if (classNode == null) {
            classNode = createNode({
              idPath: classIdPath,
              shortNameIndex: childNode.shortNameIndex,
              type: _CONTAINER_TYPES.JAVA_CLASS,
            });
            javaClassContainers.set(classIdPath, classNode);
          }

          // Adjust the dex method's short name so it starts after the "#"
          childNode.shortNameIndex = splitIndex + 1;
          TreeBuilder._attachToParent(childNode, classNode);
        } else {
          otherSymbols.push(childNode);
        }
      }

      node.children = otherSymbols;
      for (const containerNode of javaClassContainers.values()) {
        // Delay setting the parent until here so that `_attachToParent`
        // doesn't add method stats twice
        containerNode.parent = node;
        node.children.push(containerNode);
      }
    } else {
      node.children.forEach(TreeBuilder._joinDexMethodClasses);
    }
    return node;
  }

  /**
   * Formats a tree node by removing references to its desendants and ancestors.
   * This reduces how much data is sent to the UI thread at once. For large
   * trees, serialization and deserialization of the entire tree can take ~7s.
   *
   * Only children up to `depth` will be kept, and deeper children will be
   * replaced with `null` to indicate that there were children by they were
   * removed.
   *
   * Leaves with no children will always have an empty children array.
   * If a tree has only 1 child, it is kept as the UI will expand chains of
   * single children in the tree.
   *
   * Additionally sorts the formatted portion of the tree.
   * @param {TreeNode} node Node to format
   * @param {number} depth How many levels of children to keep.
   * @returns {TreeNode}
   */
  static formatNode(node, depth = 1) {
    const childDepth = depth - 1;
    // `null` represents that the children have not been loaded yet
    let children = null;
    if (depth > 0 || node.children.length <= 1) {
      // If depth is larger than 0, include the children.
      // If there are 0 children, include the empty array to indicate the node
      // is a leaf.
      // If there is 1 child, include it so the UI doesn't need to make a
      // roundtrip in order to expand the chain.
      children = node.children
          .map(n => TreeBuilder.formatNode(n, childDepth))
          .sort(_compareFunc);
    }

    return TreeBuilder._joinDexMethodClasses({
      ...node,
      children,
      parent: null,
    });
  }

  /**
   * Returns the container type for a parent node.
   * @param {string} childIdPath
   * @private
   */
  _containerType(childIdPath) {
    const useAlternateType =
      childIdPath.lastIndexOf(this._sep) > childIdPath.lastIndexOf(_PATH_SEP);
    if (useAlternateType) {
      return _CONTAINER_TYPES.COMPONENT;
    } else {
      return _CONTAINER_TYPES.DIRECTORY;
    }
  }

  /**
   * Helper to return the parent of the given node. The parent is determined
   * based in the idPath and the path seperator. If the parent doesn't yet
   * exist, one is created and stored in the parents map.
   * @param {TreeNode} childNode
   * @private
   */
  _getOrMakeParentNode(childNode) {
    // Get idPath of this node's parent.
    let parentPath;
    if (childNode.idPath === '') parentPath = _NO_NAME;
    else parentPath = dirname(childNode.idPath, this._sep);

    // check if parent exists
    let parentNode;
    if (parentPath === '') {
      // parent is root node if dirname is ''
      parentNode = this.rootNode;
    } else {
      // get parent from cache if it exists, otherwise create it
      parentNode = this._parents.get(parentPath);
      if (parentNode == null) {
        parentNode = createNode({
          idPath: parentPath,
          shortNameIndex: lastIndexOf(parentPath, this._sep) + 1,
          type: this._containerType(childNode.idPath),
        });
        this._parents.set(parentPath, parentNode);
      }
    }

    // attach node to the newly found parent
    TreeBuilder._attachToParent(childNode, parentNode);
    return parentNode;
  }

  /**
   * Iterate through every file node generated by supersize. Each node includes
   * symbols that belong to that file. Create a tree node for each file with
   * tree nodes for that file's symbols attached. Afterwards attach that node to
   * its parent directory node, or create it if missing.
   * @param {FileEntry} fileEntry File entry from data file
   */
  addFileEntry(fileEntry) {
    const idPath = this._getPath(fileEntry);
    // make node for this
    const fileNode = createNode({
      idPath,
      shortNameIndex: lastIndexOf(idPath, this._sep) + 1,
      type: _CONTAINER_TYPES.FILE,
    });
    // build child nodes for this file's symbols and attach to self
    for (const symbol of fileEntry[_KEYS.FILE_SYMBOLS]) {
      const size = symbol[_KEYS.SIZE];
      const type = symbol[_KEYS.TYPE];
      const count = symbol[_KEYS.COUNT] || 1;
      const symbolNode = createNode({
        // Join file path to symbol name with a ":"
        idPath: `${idPath}:${symbol[_KEYS.SYMBOL_NAME]}`,
        shortNameIndex: idPath.length + 1,
        size,
        type: symbol[_KEYS.TYPE],
        childStats: {[type]: {size, count}},
      });

      if (this._filterTest(symbolNode)) {
        TreeBuilder._attachToParent(symbolNode, fileNode);
      }
    }
    // unless we filtered out every symbol belonging to this file,
    if (fileNode.children.length > 0) {
      // build all ancestor nodes for this file
      let orphanNode = fileNode;
      while (orphanNode.parent == null && orphanNode !== this.rootNode) {
        orphanNode = this._getOrMakeParentNode(orphanNode);
      }
    }
  }

  /**
   * Finalize the creation of the tree and return the root node.
   */
  build() {
    this._getPath = () => '';
    this._filterTest = () => false;
    this._parents.clear();
    return this.rootNode;
  }

  /**
   * Internal handler for `find` to search for a node.
   * @private
   * @param {string[]} idPathList
   * @param {TreeNode} node
   * @returns {TreeNode | null}
   */
  _find(idPathList, node) {
    if (node == null) {
      return null;
    } else if (idPathList.length === 0) {
      // Found desired node
      return node;
    }

    const [shortNameToFind] = idPathList;
    const child = node.children.find(n => shortName(n) === shortNameToFind);

    return this._find(idPathList.slice(1), child);
  }

  /**
   * Find a node with a given `idPath` by traversing the tree.
   * @param {string} idPath
   */
  find(idPath) {
    // If `idPath` is the root's ID, return the root
    if (idPath === this.rootNode.idPath) {
      return this.rootNode;
    }

    const symbolIndex = idPath.indexOf(':');
    let path;
    if (symbolIndex > -1) {
      const filePath = idPath.slice(0, symbolIndex);
      const symbolName = idPath.slice(symbolIndex + 1);

      path = filePath.split(this._splitter);
      path.push(symbolName);
    } else {
      path = idPath.split(this._splitter);
    }

    // If the path is empty, it refers to the _NO_NAME container.
    if (path[0] === '') {
      path.unshift(_NO_NAME);
    }

    return this._find(path, this.rootNode);
  }
}

/**
 * Wrapper around fetch for requesting the same resource multiple times.
 */
class DataFetcher {
  constructor(input) {
    /** @type {AbortController | null} */
    this._controller = null;
    this.setInput(input);
  }

  /**
   * Sets the input that describes what will be fetched. Also clears the cache.
   * @param {string | Request} input URL to the resource you want to fetch.
   */
  setInput(input) {
    if (typeof this._input === 'string' && this._input.startsWith('blob:')) {
      // Revoke the previous Blob url to prevent memory leaks
      URL.revokeObjectURL(this._input);
    }

    /** @type {Uint8Array | null} */
    this._cache = null;
    this._input = input;
  }

  /**
   * Starts a new request and aborts the previous one.
   * @param {string | Request} url
   */
  async fetch(url) {
    if (this._controller) this._controller.abort();
    this._controller = new AbortController();
    return fetch(url, {
      credentials: 'same-origin',
      signal: this._controller.signal,
    });
  }

  /**
   * Yields binary chunks as Uint8Arrays. After a complete run, the bytes are
   * cached and future calls will yield the cached Uint8Array instead.
   */
  async *arrayBufferStream() {
    if (this._cache) {
      yield this._cache;
      return;
    }

    const response = await this.fetch(this._input);
    let result;
    // Use streams, if supported, so that we can show in-progress data instead
    // of waiting for the entire data file to download. The file can be >100 MB,
    // so streams ensure slow connections still see some data.
    if (response.body) {
      const reader = response.body.getReader();

      /** @type {Uint8Array[]} Store received bytes to merge later */
      let buffer = [];
      /** Total size of received bytes */
      let byteSize = 0;
      while (true) {
        // Read values from the stream
        const {done, value} = await reader.read();
        if (done) break;

        const chunk = new Uint8Array(value, 0, value.length);
        yield chunk;
        buffer.push(chunk);
        byteSize += chunk.length;
      }

      // We just cache a single typed array to save some memory and make future
      // runs consistent with the no streams mode.
      result = new Uint8Array(byteSize);
      let i = 0;
      for (const chunk of buffer) {
        result.set(chunk, i);
        i += chunk.length;
      }
    } else {
      // In-memory version for browsers without stream support
      result = new Uint8Array(await response.arrayBuffer());
      yield result;
    }

    this._cache = result;
  }

  /**
   * Transforms a binary stream into a newline delimited JSON (.ndjson) stream.
   * Each yielded value corresponds to one line in the stream.
   * @returns {AsyncIterable<Meta | FileEntry>}
   */
  async *newlineDelimtedJsonStream() {
    const decoder = new TextDecoder();
    const decoderArgs = {stream: true};
    let textBuffer = '';

    for await (const bytes of this.arrayBufferStream()) {
      if (this._controller.signal.aborted) {
        throw new DOMException('Request was aborted', 'AbortError');
      }

      textBuffer += decoder.decode(bytes, decoderArgs);
      const lines = textBuffer.split('\n');
      [textBuffer] = lines.splice(lines.length - 1, 1);

      for (const line of lines) {
        yield JSON.parse(line);
      }
    }
  }
}

/**
 * Parse the options represented as a query string, into an object.
 * Includes checks for valid values.
 * @param {string} options Query string
 */
function parseOptions(options) {
  const params = new URLSearchParams(options);

  const groupBy = params.get('group_by') || 'source_path';
  const methodCountMode = params.has('method_count');

  let minSymbolSize = Number(params.get('min_size'));
  if (Number.isNaN(minSymbolSize)) {
    minSymbolSize = 0;
  }

  const includeRegex = params.get('include');
  const excludeRegex = params.get('exclude');

  /** @type {Set<string>} */
  let typeFilter;
  if (methodCountMode) {
    typeFilter = new Set(_DEX_METHOD_SYMBOL_TYPE);
  } else {
    typeFilter = new Set(types(params.getAll(_TYPE_STATE_KEY)));
    if (typeFilter.size === 0) {
      typeFilter = new Set(_SYMBOL_TYPE_SET);
      typeFilter.delete('b');
    }
  }

  /** @type {Array<(symbolNode: TreeNode) => boolean>} */
  const filters = [];

  /** Ensure symbol size is past the minimum */
  if (minSymbolSize > 0) {
    filters.push(s => Math.abs(s.size) >= minSymbolSize);
  }

  /** Ensure the symbol size wasn't filtered out */
  if (typeFilter.size < _SYMBOL_TYPE_SET.size) {
    filters.push(s => typeFilter.has(s.type));
  }

  if (includeRegex) {
    try {
      const regex = new RegExp(includeRegex);
      filters.push(s => regex.test(s.idPath));
    } catch (err) {
      if (err.name !== 'SyntaxError') throw err;
    }
  }
  if (excludeRegex) {
    try {
      const regex = new RegExp(excludeRegex);
      filters.push(s => !regex.test(s.idPath));
    } catch (err) {
      if (err.name !== 'SyntaxError') throw err;
    }
  }

  /**
   * Check that a symbol node passes all the filters in the filters array.
   * @param {TreeNode} symbolNode
   */
  function filterTest(symbolNode) {
    return filters.every(fn => fn(symbolNode));
  }

  return {groupBy, filterTest};
}

/** @type {TreeBuilder | null} */
let builder = null;
const fetcher = new DataFetcher('data.ndjson');

/**
 * Assemble a tree when this worker receives a message.
 * @param {string} options Query string containing options for the builder.
 * @param {(msg: TreeProgress) => void} onProgress
 * @returns {Promise<TreeProgress>}
 */
async function buildTree(options, onProgress) {
  const {groupBy, filterTest} = parseOptions(options);

  /** @type {Meta | null} Object from the first line of the data file */
  let meta = null;

  /** @type {{ [gropyBy: string]: (fileEntry: FileEntry) => string }} */
  const getPathMap = {
    component(fileEntry) {
      const component = meta.components[fileEntry[_KEYS.COMPONENT_INDEX]];
      const path = getSourcePath(fileEntry);
      return `${component || '(No component)'}>${path}`;
    },
    source_path: getSourcePath,
  };

  builder = new TreeBuilder({
    sep: groupBy === 'component' ? '>' : _PATH_SEP,
    getPath: getPathMap[groupBy],
    filterTest,
  });

  /**
   * Creates data to post to the UI thread. Defaults will be used for the root
   * and percent values if not specified.
   * @param {{root?:TreeNode,percent?:number,error?:Error}} data Default data
   * values to post.
   */
  function createProgressMessage(data = {}) {
    let {percent} = data;
    if (percent == null) {
      if (meta == null) {
        percent = 0;
      } else {
        percent = Math.max(builder.rootNode.size / meta.total, 0.1);
      }
    }

    const message = {
      root: TreeBuilder.formatNode(data.root || builder.rootNode),
      percent,
      diffMode: meta && meta.diff_mode,
    };
    if (data.error) {
      message.error = data.error.message;
    }
    return message;
  }

  /**
   * Post data to the UI thread. Defaults will be used for the root and percent
   * values if not specified.
   */
  function postToUi() {
    const message = createProgressMessage();
    message.id = 0;
    onProgress(message);
  }

  try {
    // Post partial state every second
    let lastBatchSent = Date.now();
    for await (const dataObj of fetcher.newlineDelimtedJsonStream()) {
      if (meta == null) {
        // First line of data is used to store meta information.
        meta = /** @type {Meta} */ (dataObj);
        postToUi();
      } else {
        builder.addFileEntry(/** @type {FileEntry} */ (dataObj));
        const currentTime = Date.now();
        if (currentTime - lastBatchSent > 500) {
          postToUi();
          await Promise.resolve();  // Pause loop to check for worker messages
          lastBatchSent = currentTime;
        }
      }
    }

    return createProgressMessage({
      root: builder.build(),
      percent: 1,
    });
  } catch (error) {
    if (error.name === 'AbortError') {
      console.info(error.message);
    } else {
      console.error(error);
    }
    return createProgressMessage({error});
  }
}

const actions = {
  /** @param {{input:string,options:string}} data */
  load(data) {
    if (data.input) fetcher.setInput(data.input);
    return buildTree(data.options, progress => {
      // @ts-ignore
      self.postMessage(progress);
    });
  },
  /** @param {string} path */
  async open(path) {
    if (!builder) throw new Error('Called open before load');
    const node = builder.find(path);
    return TreeBuilder.formatNode(node);
  },
};

/**
 * Call the requested action function with the given data. If an error is thrown
 * or rejected, post the error message to the UI thread.
 * @param {number} id Unique message ID.
 * @param {string} action Action type, corresponding to a key in `actions.`
 * @param {any} data Data to supply to the action function.
 */
async function runAction(id, action, data) {
  try {
    const result = await actions[action](data);
    // @ts-ignore
    self.postMessage({id, result});
  } catch (err) {
    // @ts-ignore
    self.postMessage({id, error: err.message});
    throw err;
  }
}

const runActionDebounced = debounce(runAction, 0);

/**
 * @param {MessageEvent} event Event for when this worker receives a message.
 */
self.onmessage = async event => {
  const {id, action, data} = event.data;
  if (action === 'load') {
    // Loading large files will block the worker thread until complete or when
    // an await statement is reached. During this time, multiple load messages
    // can pile up due to filters being adjusted. We debounce the load call
    // so that only the last message is read (the current set of filters).
    runActionDebounced(id, action, data);
  } else {
    runAction(id, action, data);
  }
};
