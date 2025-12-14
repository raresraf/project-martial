#include <bits/stdc++.h>

using namespace std;

int main() {
  int ans = 0;
  string inp;
  map<char, int> mapa;
  cin >> inp;
  for (int i = 0; i < inp.length(); i++) {
    mapa[inp[i]]++;
    if (mapa[inp[i]] == 1)
      ans++;
  }
  if (ans % 2 == 0)
    printf("CHAT WITH HER!");
  else
    printf("IGNORE HIM!");
  return 0;
}