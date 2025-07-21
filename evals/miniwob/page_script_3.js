var MultimodalWebSurfer = MultimodalWebSurfer || (function() {
    let nextLabel = 10;
    
    let annotateXPath = function(element) {
        // if (!element) return;
        // if (element.id) {
        //     return `//${element.tagName.toLowerCase()}[@id="${element.id}"]`;
        // } else if (element.tagName.toLowerCase() === 'option') {
        //     let selectElem = element.closest('select');
        //     let optionIndex = Array.from(selectElem.options).indexOf(element);
        //     let selectXPath = annotateXPath(selectElem);
        //     return `${selectXPath}/option[${optionIndex + 1}]`;
        // }
        // else {
        //     let siblings = Array.from(element.parentNode.children).filter(e => e.tagName === element.tagName);
        //     let siblingIndex = siblings.indexOf(element) + 1;
        //     let parentXPath = annotateXPath(element.parentNode);
        //     return `${parentXPath}/${element.tagName.toLowerCase()}[${siblingIndex}]`;
        // }

        if (element.id !== "") {
            return `//${element.tagName.toLowerCase()}[@id="${element.id}"]`;
        } else if (element.tagName.toLowerCase() === "option" && element.innerText !== "") {
            return `//${element.tagName.toLowerCase()}[text()="${element.innerText}"]`;
        } else if (element.innerText !== "") {
            return `//${element.tagName.toLowerCase()}[text()="${element.innerText}"]`;
        } else {
            // Build the XPath based on the hierarchy of the element.
            const paths = [];
    
            while (element && element.nodeType === Node.ELEMENT_NODE) {
                let index = 0;
                let sibling = element.previousSibling;
    
                // Count the index of the element relative to its siblings of the same type
                while (sibling) {
                    if (sibling.nodeType === Node.ELEMENT_NODE && sibling.tagName === element.tagName) {
                        index++;
                    }
                    sibling = sibling.previousSibling;
                }
    
                // Add the current element's part to the path
                const tagName = element.tagName.toLowerCase();
                const pathIndex = (index ? `[${index + 1}]` : ''); // Add index if necessary
                paths.unshift(`${tagName}${pathIndex}`);
    
                // Move up to the parent node
                element = element.parentNode;
            }
    
            return `//${paths.join("/")}`;
        }
        
        
        // var allNodes = document.getElementsByTagName('*'); 
        // for (var segs = []; elm && elm.nodeType == 1; elm = elm.parentNode) 
        // { 
        //     if (elm.hasAttribute('id')) { 
        //             var uniqueIdCount = 0; 
        //             for (var n=0;n < allNodes.length;n++) { 
        //                 if (allNodes[n].hasAttribute('id') && allNodes[n].id == elm.id) uniqueIdCount++; 
        //                 if (uniqueIdCount > 1) break; 
        //             }; 
        //             if ( uniqueIdCount == 1) { 
        //                 segs.unshift('id("' + elm.getAttribute('id') + '")'); 
        //                 return segs.join('/'); 
        //             } else { 
        //                 segs.unshift(elm.localName.toLowerCase() + '[@id="' + elm.getAttribute('id') + '"]'); 
        //             } 
        //     } else if (elm.hasAttribute('class')) { 
        //         segs.unshift(elm.localName.toLowerCase() + '[@class="' + elm.getAttribute('class') + '"]'); 
        //     } else { 
        //         for (i = 1, sib = elm.previousSibling; sib; sib = sib.previousSibling) { 
        //             if (sib.localName == elm.localName)  i++; }; 
        //             segs.unshift(elm.localName.toLowerCase() + '[' + i + ']'); 
        //     }; 
        // }; 
        // return segs.length ? '/' + segs.join('/') : null; 
    };

    let roleMapping = {
        "a": "link",
        "area": "link",
        "button": "button",
        "input, type=button": "button",
        "input, type=checkbox": "checkbox",
        "input, type=email": "textbox",
        "input, type=number": "spinbutton",
        "input, type=radio": "radio",
        "input, type=range": "slider",
        "input, type=reset": "button",
        "input, type=search": "searchbox",
        "input, type=submit": "button",
        "input, type=tel": "textbox",
        "input, type=text": "textbox",
        "input, type=url": "textbox",
        "search": "search",
        "select": "combobox",
        "option": "option",
        "textarea": "textbox"
    };
  
    let getCursor = function(elm) {
        return window.getComputedStyle(elm)["cursor"];
    };
  
    // let getInteractiveElements = function() {
    //     let results = [];
    //     let roles = ["scrollbar", "searchbox", "slider", "spinbutton", "switch", "tab", "treeitem", "button", "checkbox", "gridcell", "link", "menuitem", "menuitemcheckbox", "menuitemradio", "option", "progressbar", "radio", "textbox", "combobox", "menu", "tree", "treegrid", "grid", "listbox", "radiogroup", "widget"];
    
    //     // Get the main interactive elements
    //     let nodeList = document.querySelectorAll("input, select, textarea, button, [href], [onclick], [contenteditable], [tabindex]:not([tabindex='-1'])");
    //     for (let i = 0; i < nodeList.length; i++) {
    //         let node = nodeList[i];
    //         let computedStyle = window.getComputedStyle(node);
    //         if (!(node.disabled || computedStyle.pointerEvents === 'none')) {
    //             results.push(node);
    //         }
    //     }
    
    //     // Anything not already included that has a suitable role
    //     nodeList = document.querySelectorAll("[role]");
    //     for (let i = 0; i < nodeList.length; i++) {
    //         let node = nodeList[i];
    //         if (results.indexOf(node) === -1) {
    //             let role = node.getAttribute("role");
    //             let computedStyle = window.getComputedStyle(node);
    //             if (roles.indexOf(role) > -1 && !(node.disabled || computedStyle.pointerEvents === 'none')) {
    //                 results.push(node);
    //             }
    //         }
    //     }
    
    //     // Any element that changes the cursor to something implying interactivity
    //     nodeList = document.querySelectorAll("*");
    //     for (let i = 0; i < nodeList.length; i++) {
    //         let node = nodeList[i];
    //         let computedStyle = window.getComputedStyle(node);
    //         // Cursor suggests interactivity and pointer events are not none
    //         if (!(computedStyle.cursor === 'auto' || computedStyle.cursor === 'default' || computedStyle.cursor === 'none' || computedStyle.cursor === 'text' || computedStyle.cursor === 'vertical-text' || computedStyle.cursor === 'not-allowed' || computedStyle.cursor === 'no-drop' || computedStyle.pointerEvents === 'none')) {
    //             if (results.indexOf(node) === -1) {
    //                 results.push(node);
    //             }
    //         }
    //     }
    
    //     return results;
    // };

    let getInteractiveElements = function() {

        let results = []
        let roles = ["scrollbar", "searchbox", "slider", "spinbutton", "switch", "tab", "treeitem", "button", "checkbox", "gridcell", "link", "menuitem", "menuitemcheckbox", "menuitemradio", "option", "progressbar", "radio", "textbox", "combobox", "menu", "tree", "treegrid", "grid", "listbox", "radiogroup", "widget"];
        let inertCursors = ["auto", "default", "none", "text", "vertical-text", "not-allowed", "no-drop"];
  
        // Get the main interactive elements
        // let nodeList = document.querySelectorAll("div, input, select, textarea, button, [href], [onclick], [contenteditable], [tabindex]:not([tabindex='-1'])");
        // let nodeList = document.querySelectorAll("*");
        let nodeList = document.querySelectorAll(":not(html):not(head):not(body)");
        
        for (let i=0; i<nodeList.length; i++) { // Copy to something mutable
            results.push(nodeList[i]);
        }
        // console.log("nodeList 1", nodeList);

        // Anything not already included that has a suitable role
        nodeList = document.querySelectorAll("[role]");
        for (let i=0; i<nodeList.length; i++) { // Copy to something mutable
            if (results.indexOf(nodeList[i]) == -1) {
                let role = nodeList[i].getAttribute("role");
            if (roles.indexOf(role) > -1) {
                    results.push(nodeList[i]);
            }
        }
        }
  
        // Any element that changes the cursor to something implying interactivity
        nodeList = document.querySelectorAll("*");
        for (let i=0; i<nodeList.length; i++) {
           let node = nodeList[i];
  
           // Cursor is default, or does not suggest interactivity
           let cursor = getCursor(node);
           if (inertCursors.indexOf(cursor) >= 0) {
               continue;
           }
  
           // Move up to the first instance of this cursor change
           parent = node.parentNode;
           while (parent && getCursor(parent) == cursor) {
               node = parent;
           parent = node.parentNode;
           }
  
           // Add the node if it is new
           if (results.indexOf(node) == -1) {
               results.push(node);
           }
        }
        
        // console.log("nodeList 2", results);

        return results;
    };
  
    let labelElements = function(elements) {
        for (let i=0; i<elements.length; i++) {
            if (!elements[i].hasAttribute("__elementId")) {
                elements[i].setAttribute("__elementId", "" + (nextLabel++));
            }
            // Annotate XPath
            // elements[i].setAttribute("xpath", annotateXPath(elements[i]));
        }
    };
  
    let isTopmost = function(element, x, y) {
       let hit = document.elementFromPoint(x, y);
  
       // Hack to handle elements outside the viewport
       if (hit === null) {
           return true; 
       }
  
       while (hit) {
           if (hit == element) return true;
           hit = hit.parentNode;
       }
       return false;
    };
  
    let getFocusedElementId = function() {
       let elm = document.activeElement;
       while (elm) {
           if (elm.hasAttribute && elm.hasAttribute("__elementId")) {
           return elm.getAttribute("__elementId");
       }
           elm = elm.parentNode;
       }
       return null;
    };
  
    // let trimmedInnerText = function(element) {
    //     if (!element) {
    //         return "";
    //     }
    //     let text = element.innerText;
    //     console.log("trimmedInnerText", text);

    //     if (!text) {
    //         return "";
    //     }
    //     return text.trim();
    // };

    let trimmedInnerText = function(element) {
        if (!element) {
            return "";
        }
    
        // Clone the element
        let clone = element.cloneNode(true);
    
        // Remove all child elements
        Array.from(clone.children).forEach(child => clone.removeChild(child));
    
        // Get and trim the text content
        let text = clone.textContent;
        // console.log("trimmedInnerText (own text):", text);
    
        return text ? text.trim() : "";
    };
  
    let getApproximateAriaName = function(element) {
        // Check for aria labels
        if (element.hasAttribute("aria-labelledby")) {
            let buffer = "";
        let ids = element.getAttribute("aria-labelledby").split(" ");
        for (let i=0; i<ids.length; i++) {
                let label = document.getElementById(ids[i]);
            if (label) {
                buffer = buffer + " " + trimmedInnerText(label);
                }
            }
        return buffer.trim();
        }
  
        if (element.hasAttribute("aria-label")) {
        return element.getAttribute("aria-label");
        }
  
        // Check for labels
        if (element.hasAttribute("id")) {
            let label_id = element.getAttribute("id");
            let label = "";
            let labels = document.querySelectorAll("label[for='" + label_id + "']");
            for (let j=0; j<labels.length; j++) {
                label += labels[j].innerText + " ";
            }
            label = label.trim();
            if (label != "") {
                return label;
            }
        }
  
        if (element.parentElement && element.parentElement.tagName == "LABEL") {
            return element.parentElement.innerText;
        }
  
        // Check for alt text or titles
        if (element.hasAttribute("alt")) {
        return element.getAttribute("alt")
        }
  
        if (element.hasAttribute("title")) {
        return element.getAttribute("title")
        }
  
        return trimmedInnerText(element);
        // return "";
    };
  
    let getApproximateAriaRole = function(element) {
        let tag = element.tagName.toLowerCase();
        if (tag == "input" && element.hasAttribute("type")) {
            tag = tag + ", type=" + element.getAttribute("type");
        }
        if (element.hasAttribute("class")) {
            // console.log("class present");
            tag = tag + ", class=" + element.getAttribute("class");
        }
        // else {
            // console.log("class absent");
        // }

        // data-color="orange" style="background-color: orange;"
        if (element.hasAttribute("data-color")) {
            tag = tag + ", data-color=" + element.getAttribute("data-color");
        }
  
        if (element.hasAttribute("role")) {
            return [element.getAttribute("role"), tag];
        }
        else if (tag in roleMapping) {
            return [roleMapping[tag], tag];
        }
        else {
        return ["", tag];
        }
    };
  
    let getInteractiveRects = function() {
        labelElements(getInteractiveElements());

        let elements = document.querySelectorAll("[__elementId]");
        // console.log("elements", elements);

        let results = {};
        for (let i=0; i<elements.length; i++) {
           let key = elements[i].getAttribute("__elementId");
           let rects = elements[i].getClientRects();
       let ariaRole = getApproximateAriaRole(elements[i]);
       let ariaName = getApproximateAriaName(elements[i]);
       let vScrollable = elements[i].scrollHeight - elements[i].clientHeight >= 1;
  
       let record = {
               "tag_name": ariaRole[1],
           "role": ariaRole[0],
           "aria-name": ariaName,
           "v-scrollable": vScrollable,
           "rects": [],
           "xpath": annotateXPath(elements[i])
       };
  
           for (const rect of rects) {
           let x = rect.left + rect.width/2;
               let y = rect.top + rect.height/2;
               if (isTopmost(elements[i], x, y)) {
           record["rects"].push(JSON.parse(JSON.stringify(rect)));
               }
           }
  
    //    if (record["rects"].length > 0) {
               results[key] = record;
            //    console.log("record accept", record);
        //    }
        //    else{
            // console.log("record reject", record);
        // }
        }
        
        return results;
    };
  
    let getVisualViewport = function() {
        let vv = window.visualViewport;
        let de = document.documentElement;
        return {
            "height":     vv ? vv.height : 0,
        "width":      vv ? vv.width : 0,
        "offsetLeft": vv ? vv.offsetLeft : 0,
        "offsetTop":  vv ? vv.offsetTop : 0,
        "pageLeft":   vv ? vv.pageLeft  : 0,
        "pageTop":    vv ? vv.pageTop : 0,
        "scale":      vv ? vv.scale : 0,
        "clientWidth":  de ? de.clientWidth : 0,
        "clientHeight": de ? de.clientHeight : 0,
        "scrollWidth":  de ? de.scrollWidth : 0,
        "scrollHeight": de ? de.scrollHeight : 0
        };
    };
  
    let _getMetaTags = function() {
        let meta = document.querySelectorAll("meta");
        let results = {};
        for (let i = 0; i<meta.length; i++) {
            let key = null;
            if (meta[i].hasAttribute("name")) {
                key = meta[i].getAttribute("name");
            }
            else if (meta[i].hasAttribute("property")) {
                key = meta[i].getAttribute("property");
            }
            else {
                continue;
            }
            if (meta[i].hasAttribute("content")) {
                results[key] = meta[i].getAttribute("content");
            }
        }
        return results;
    };
  
    let _getJsonLd = function() {
        let jsonld = [];
        let scripts = document.querySelectorAll('script[type="application/ld+json"]');
        for (let i=0; i<scripts.length; i++) {
            jsonld.push(scripts[i].innerHTML.trim());
        }
        return jsonld;
     };
  
     // From: https://www.stevefenton.co.uk/blog/2022/12/parse-microdata-with-javascript/
     let _getMicrodata = function() {
        function sanitize(input) {
            return input.replace(/\s/gi, ' ').trim();
        }
  
        function addValue(information, name, value) {
            if (information[name]) {
                if (typeof information[name] === 'array') {
                    information[name].push(value);
                } else {
                    const arr = [];
                    arr.push(information[name]);
                    arr.push(value);
                    information[name] = arr;
                }
            } else {
                information[name] = value;
            }
        }
  
        function traverseItem(item, information) {
           const children = item.children;
          
           for (let i = 0; i < children.length; i++) {
               const child = children[i];
  
               if (child.hasAttribute('itemscope')) {
                   if (child.hasAttribute('itemprop')) {
                       const itemProp = child.getAttribute('itemprop');
                       const itemType = child.getAttribute('itemtype');
  
                       const childInfo = {
                           itemType: itemType
                       };
  
                       traverseItem(child, childInfo);
  
                       itemProp.split(' ').forEach(propName => {
                           addValue(information, propName, childInfo);
                       });
                   }
  
               } else if (child.hasAttribute('itemprop')) {
                   const itemProp = child.getAttribute('itemprop');
                   itemProp.split(' ').forEach(propName => {
                       if (propName === 'url') {
                           addValue(information, propName, child.href);
                       } else {
                           addValue(information, propName, sanitize(child.getAttribute("content") || child.content || child.textContent || child.src || ""));
                       }
                   });
                   traverseItem(child, information);
               } else {
                   traverseItem(child, information);
               }
           }
        }
  
        const microdata = [];
  
        document.querySelectorAll("[itemscope]").forEach(function(elem, i) {
           const itemType = elem.getAttribute('itemtype');
           const information = {
               itemType: itemType
           };
           traverseItem(elem, information);
           microdata.push(information);
        });
      
        return microdata;
     };
  
     let getPageMetadata = function() {
         let jsonld = _getJsonLd();
         let metaTags = _getMetaTags();
         let microdata = _getMicrodata();
         let results = {}
         if (jsonld.length > 0) {
             try {
                 results["jsonld"] = JSON.parse(jsonld);
             } 
         catch (e) {
                 results["jsonld"] = jsonld;
         }
         }
         if (microdata.length > 0) {
         results["microdata"] = microdata;
         }
         for (let key in metaTags) {
         if (metaTags.hasOwnProperty(key)) {
             results["meta_tags"] = metaTags;
             break;
             }
         }
         return results;
     };	
  
     return {
         getInteractiveRects: getInteractiveRects,
         getVisualViewport: getVisualViewport,
         getFocusedElementId: getFocusedElementId,
         getPageMetadata: getPageMetadata,
     };
  })();