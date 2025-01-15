#include <cctype>
#include <cstdio>
#include <cstring>
#include <iostream>

using namespace std;

int main() {
  int n, temp1, temp2;
  int pos = 0;
  int arr[5][5];
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> n;
      arr[i][j] = n;
      if (n == 1) {
        temp1 = i;
        temp2 = j;
      }
    }
  }

  cout << abs(temp1 - 2) + abs(temp2 - 2);

  return 0;
}
