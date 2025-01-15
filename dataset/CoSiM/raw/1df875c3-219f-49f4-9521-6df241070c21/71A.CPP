#include <cstdio>
#include <cstring>
char x[101];

using namespace std;

int main() {
  int n, i;
  scanf("%d", &n);
  for (i = 0; i < n; i++) {
    scanf("%s", &x);
    if (strlen(x) > 10) {
      if (!strcmp(x, "localization")) {
        printf("l10n\n");
      } else if (!strcmp(x, "internationalization")) {
        printf("i18n\n");
      } else {
        printf("%c", x[0]);
        printf("%d", strlen(x) - 2);
        printf("%c\n", x[strlen(x) - 1]);
      }
    } else
      printf("%s\n", x);
  }
  return 0;
}
