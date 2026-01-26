#include <bits/stdc++.h>

using namespace std;

int main() {

  string s1, s2;

  cin >> s1;
  cin >> s2;

  for (int i = 0; i < s1.length(); ++i) {

    s1[i] = tolower(s1[i]);
    s2[i] = tolower(s2[i]);
  }

  if (s1 > s2)
    printf("1\n");
  else if (s1 == s2)
    printf("0\n");
  else
    printf("-1\n");

  return 0;
}