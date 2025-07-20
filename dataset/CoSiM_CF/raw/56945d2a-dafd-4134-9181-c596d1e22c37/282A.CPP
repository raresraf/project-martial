#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, bit = 0;
  scanf("%d", &n);
  string op;
  for (int i = 0; i < n; ++i) {
    cin >> op;
    if (op.find("++") != string::npos)
      bit++;
    else
      bit--;
  }
  printf("%d\n", bit);
}