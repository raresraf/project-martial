#include <bits/stdc++.h>
#include <string>

using namespace std;
int main() {
  char kata[101];
  scanf("%s", &kata);
  int len = strlen(kata);
  for (int i = 0; i < len; i++) {
    char x = tolower(kata[i]);
    if (x == 'a' || x == 'e' || x == 'o' || x == 'i' || x == 'u') {
      continue;
    }
    printf(".%c", x);
  }
  printf("\n");
}