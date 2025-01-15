#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char const *argv[]) {
  char a[100] = {0}, b[100] = {0}, c;
  int i = 0;

  while (true) {
    scanf("%c", &c);
    if (c == '\n')
      break;
    if (c > 'Z')
      c -= 32;
    a[i] = c;
    i++;
  }
  i = 0;
  while (true) {
    scanf("%c", &c);
    if (c == '\n')
      break;
    if (c > 'Z')
      c -= 32;
    b[i] = c;
    i++;
  }

  for (int j = 0; j < 100; j++) {
    if (a[j] == 0 && b[j] == 0) {
      cout << 0;
      break;
    }
    if (a[j] > b[j]) {
      cout << 1;
      break;
    }
    if (a[j] < b[j]) {
      cout << -1;
      break;
    }
  }
  return 0;
}
