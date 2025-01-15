/*
 *	User: Isanchez_Aguilar
 *	Problem: CodeForces 112A - Petya and Strings
 */
#include <bits/stdc++.h>

using namespace std;

int main(void) {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  char str[101];
  scanf("%s\n", str);
  int length = strlen(str);
  int comparison = 0;
  for (int i = 0; i < length; ++i) {
    char letter = tolower(getchar());
    if (!comparison) {
      char c = tolower(str[i]);
      if (letter < c)
        comparison = 1;
      else if (letter > c)
        comparison = -1;
    }
  }
  printf("%d\n", comparison);
  return 0;
}