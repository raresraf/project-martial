#include <bits/stdc++.h>
using namespace std;

int main() {
  char str[100];

  scanf("%[^\n]", str);
  int n = strlen(str);
  for (int i = 0; i < n; i++) {
    int k = str[i];
    if (k == 65 || k == 69 || k == 73 || k == 79 || k == 85 || k == 89 ||
        k == 97 || k == 101 || k == 105 || k == 111 || k == 117 || k == 121)
      continue;
    else if (k < 97) {
      char c;
      k = k + 32;
      c = k;
      cout << '.' << c;
    } else
      cout << '.' << str[i];
  }
  cout << endl;

  return 0;
}
