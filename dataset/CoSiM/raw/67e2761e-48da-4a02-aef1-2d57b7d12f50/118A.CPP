/*
 *	User: Isanchez_Aguilar
 *	Problem: CodeForces 118A - String Task
 */
#include <bits/stdc++.h>

using namespace std;

int main(void) {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  char str[101];
  scanf("%s", str);
  int length = strlen(str);
  for (int i = 0; i < length; ++i) {
    char letter = tolower(str[i]);
    if (letter != 'a' && letter != 'e' && letter != 'i' && letter != 'o' &&
        letter != 'u' && letter != 'y')
      printf(".%c", letter);
  }
  return 0;
}