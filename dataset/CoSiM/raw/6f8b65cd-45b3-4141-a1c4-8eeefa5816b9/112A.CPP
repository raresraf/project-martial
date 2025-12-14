#include <bits/stdc++.h>

using namespace std;

string toLower(string w) {
  for (int i = 0; i < w.length(); i++)
    if (w[i] >= 'A' && w[i] <= 'Z')
      w[i] += 32;
  return w;
}

int main() {
  string a, b;
  cin >> a >> b;
  a = toLower(a);
  b = toLower(b);
  if (a < b)
    printf("-1");
  if (b < a)
    printf("1");
  if (a == b)
    printf("0");
  return 0;
}