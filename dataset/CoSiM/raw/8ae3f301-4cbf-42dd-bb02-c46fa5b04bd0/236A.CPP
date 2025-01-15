#include <algorithm>
#include <iostream>
#include <string.h>

using namespace std;

int main() {
  string a;
  cin >> a;
  sort(a.begin(), a.end());
  int count{};
  for (int i = 0; i < a.length() - 1; i++) {
    if (a[i] == a[i + 1])
      count++;
  }
  int value{-count};
  value += a.length();
  if (value % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";

  return 0;
}