#include <iostream>
using namespace std;

int main() {
  int x, i, remove = 0;

  cin >> x;
  char stone[x];

  for (i = 0; i < x; i++)
    cin >> stone[i];

  for (i = 0; i < x; i++) {
    if (stone[i] == stone[i + 1])
      remove++;
    else
      continue;
  }

  cout << remove;
  return 0;
}
