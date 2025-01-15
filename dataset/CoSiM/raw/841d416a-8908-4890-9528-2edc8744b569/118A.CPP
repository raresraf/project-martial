// https://codeforces.com/problemset/problem/118/A
// SOLUTION BY DN54321

#include <ctype.h>
#include <iostream>
#include <string.h>
#include <vector>
typedef long long ll;
using namespace std;
int main() {
  char s[101];
  char s1[301];
  int p = 0;
  char vowels[6] = {'a', 'e', 'i', 'o', 'u', 'y'}; // y is not a vowel =.=
  scanf("%s", &s);
  for (int i = 0; i < strlen(s); ++i) {
    bool isvowel = 0;
    for (int j = 0; j < 6; ++j) {
      if (tolower(s[i]) == vowels[j]) {
        isvowel = 1;
        break;
      }
    }
    if (!isvowel) {
      s1[p++] = '.';
      s1[p++] = tolower(s[i]);
    }
  }
  s1[p] = '\0';
  printf("%s\n", s1);
  return 0;
}