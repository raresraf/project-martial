#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  string s;
  cin >> s;
  char array[200];
  int count = 0;
  for (int i = 0; i <= s.size() / 2; i++) {
    array[i] = s[2 * i];
    count++;
  }

  sort(array, array + count);
  for (int i = 0; i < count; i++) {
    if (i == count - 1)
      cout << array[i] << endl;
    else
      cout << array[i] << "+";
  }
  return 0;
}
