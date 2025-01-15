#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string.h>

using namespace std;
int main() {
  int c = 0;
  string name;
  cin >> name;
  sort(name.begin(), name.end());

  for (int i = 0; i < (int)name.length(); i++) {
    if (name[i] == name[i + 1])
      ;
    else
      c++;
  }

  if (c++ % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}
