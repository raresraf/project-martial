/**
 * Created by LonelyEnvoy on 2017-7-23.
 * A. Way Too Long Words
 * Keywords: simple strings
 */

#include <cstdio>
#include <cstring>
using namespace std;

int main() {
  int n;
  char buffer[101];
  int len;
  scanf("%d", &n);
  while (n--) {
    scanf("%s", buffer);
    len = strlen(buffer);
    if (len > 10) {
      printf("%c", buffer[0]);
      printf("%d", len - 2);
      printf("%c", buffer[len - 1]);
    } else {
      printf("%s", buffer);
    }
    printf("\n");
  }
}
