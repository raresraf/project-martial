#include <iostream>

using namespace std;

int main() {
  int a;
  int i = -1;
  do {
    cin >> a;
    ++i;
  } while (!a);
  int count = abs(2 - i / 5) + abs(2 - i % 5);
  printf("%d", count);
}