#include <iostream>
using namespace std;

int main() {
  int a, i, j;
  int c = 0;
  int d = 0;
  int p[2];
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> a;
      if (a == 1) {
        p[0] = i;
        p[1] = j;
        break;
      }
    }
  }
  if (p[0] > 3) {
    while (p[0] != 3) {
      p[0]--;
      c++;
    }
  }

  else {
    while (p[0] != 3) {
      p[0]++;
      c++;
    }
  }

  if (p[1] > 3) {
    while (p[1] != 3) {
      p[1]--;
      c++;
    }
  }

  else {
    while (p[1] != 3) {
      p[1]++;
      c++;
    }
  }

  cout << c;

  return 0;
}