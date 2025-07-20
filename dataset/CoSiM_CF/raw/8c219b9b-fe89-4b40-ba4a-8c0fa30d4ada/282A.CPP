#include <iostream>
using namespace std;
int main() {
  int n, f, num = 0;
  char c;
  scanf("%d", &n);

  while (n) {
    getchar();
    c = getchar();
    if (c == '+') {
      f = 1;
    } else if (c == '-') {
      f = 0;
    }
    c = getchar();
    if (c == '+') {
      f = 1;
    } else if (c == '-') {
      f = 0;
    }
    c = getchar();
    if (c == '+') {
      f = 1;
    } else if (c == '-') {
      f = 0;
    }
    if (f == 0) {
      num--;
    } else {
      num++;
    }
    n--;
  }
  cout << num;
}