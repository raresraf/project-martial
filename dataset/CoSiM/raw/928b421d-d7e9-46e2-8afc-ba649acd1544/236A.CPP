/* Date: 09.01.16
Problem: 236A - Boy or Girl
*/
#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  char c[100];
  scanf("%s", c);

  int l = strlen(c);

  set<char> myset;

  for (int i = 0; i < l; i++) {
    myset.insert(c[i]);
  }

  if (myset.size() % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else
    cout << "IGNORE HIM!" << endl;

  return 0;
}
