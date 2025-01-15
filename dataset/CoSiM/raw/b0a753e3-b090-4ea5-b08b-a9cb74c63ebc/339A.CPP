#include <cstring>
#include <iostream>
using namespace std;
int main() {
  char a[100];
  int b[3] = {0};
  int i = 0, j = 0, z;
  fgets(a, sizeof(a), stdin);
  for (int i = 0; i < strlen(a); i += 2) {
    if (a[i] == '1')
      b[0]++;
    else if (a[i] == '2')
      b[1]++;
    else
      b[2]++;
  }
  j = b[0] + b[1] + b[2];
  i = 0;
  while (b[0] != 0 || b[1] != 0 || b[2] != 0) {
    if (b[0] != 0) {
      cout << 1;
      b[0]--;
    } else if (b[1] != 0) {
      cout << 2;
      b[1]--;
    } else if (b[2] != 0) {
      cout << 3;
      b[2]--;
    }
    if (i < j - 1) {
      cout << '+';
      i++;
    }
  }
  return 0;
}