#include <cstring>
#include <iostream>
using namespace std;

int main() {
  char a[4];
  int x = 0, n = 0;

  cin >> n;

  for (int i = 0; i < n; i++) {
    cin >> a;

    if (strcmp(a, "X++") == 0)
      x++;
    else if (strcmp(a, "X--") == 0)
      x--;
    else if (strcmp(a, "++X") == 0)
      ++x;
    else if (strcmp(a, "--X") == 0)
      --x;
  }
  cout << x;
}
