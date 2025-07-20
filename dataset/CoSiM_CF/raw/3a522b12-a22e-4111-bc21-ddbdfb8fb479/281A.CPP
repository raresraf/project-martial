#include <bits/stdc++.h>
using namespace std;
int main() {
  char name[10000];
  cin >> name;
  if (name[0] >= 97 && name[0] <= 122) {
    name[0] = name[0] - 32;
  }
  cout << name;
  return 0;
}
