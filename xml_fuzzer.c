#include <libxml/parser.h>
#include <libxml/tree.h>
#include <stdint.h>
#include <stddef.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    xmlDocPtr doc = xmlReadMemory((const char *)data, size,
                                  "noname.xml", NULL, 0);
    if (doc != NULL) {
        xmlFreeDoc(doc);
    }
    return 0;
}
