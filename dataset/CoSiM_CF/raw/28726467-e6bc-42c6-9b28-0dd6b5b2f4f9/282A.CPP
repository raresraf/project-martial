#include <bits/stdc++.h>

using namespace std;

inline void inp(int &n) { // fast input function
  n = 0;
  int ch = getchar(), sign = 1;
  while (ch < '0' || ch > '9') {
    if (ch == '-')
      sign = -1;
    ch = getchar();
  }
  while (ch >= '0' && ch <= '9')
    n = (n << 3) + (n << 1) + ch - '0', ch = getchar();
  n = n * sign;
}

int main() {
  int n;
  inp(n);
  int x = 0;

  while (n--) {
    string str;
    getline(cin, str);
    // printf("%s\n",str.c_str());
    if (str.find("+") != string::npos) {
      x++;
    } else if (str.find("-") != string::npos) {
      x--;
    }
  }
  printf("%d\n", x);
}