#include <bits/stdc++.h>

using namespace std;

int main() {
  string s1, s2;
  getline(cin, s1);
  getline(cin, s2);
  for (int i = 0; i < s1.size(); i++) {
    if (s1[i] >= 65 && s1[i] <= 91) {
      s1[i] = s1[i] + 32;
    } else {
      s1[i] = s1[i];
    }

    if (s2[i] >= 65 && s2[i] <= 91) {
      s2[i] = s2[i] + 32;
    } else {
      s2[i] = s2[i];
    }
  }

  if (s1.compare(s2) > 0) {
    printf("1\n");
  } else if (s1.compare(s2) < 0) {
    printf("-1\n");
  } else {
    printf("0\n");
  }

  return 0;
}
