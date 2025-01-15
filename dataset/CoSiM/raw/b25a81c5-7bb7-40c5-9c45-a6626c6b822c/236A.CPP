#include <cctype>
#include <iostream>
#include <string>

using namespace std;

int main() {
  // freopen("input.txt", "r", stdin);
  bool a['z' + 1] = {};
  char c;
  while (cin >> c)
    a[c] = true;
  int n = 0;
  for (char c = 'a'; c <= 'z'; c++)
    if (a[c])
      n++;
  if (n & 1)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";

  return 0;
}
