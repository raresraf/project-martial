#include <bits/stdc++.h>
#include <cstring>

using namespace std;
int main() {
  char str[100];
  int n;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> setw(101) >> str;
    int len = strlen(str);
    if (strlen(str) > 10) {
      printf("%c%d%c\n", str[0], len - 2, str[len - 1]);
    } else {
      puts(str);
    }
  }
}
