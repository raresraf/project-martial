#include <bits/stdc++.h>

using namespace std;

int main() {
  unordered_set<char> c;
  char *p = (char *)malloc(sizeof(char) * 100);
  cin >> p;
  while (*p) {
    c.insert(*p);
    *p++;
  }
  if (c.size() % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNOR HIM!";
  return 0;
}
