#include <iostream>
#include <string>

using namespace std;

int main() {
  char c;
  int a[4] = {};
  while (cin >> c)
    if (c != '+')
      a[c - '0']++;
  for (int i = 1; i <= 3; i++)
    if (a[i]) {
      cout << i;
      a[i]--;
      for (int j = 1; j <= 3; j++)
        while (a[j]) {
          cout << '+' << j;
          a[j]--;
        }
    }

  return 0;
}