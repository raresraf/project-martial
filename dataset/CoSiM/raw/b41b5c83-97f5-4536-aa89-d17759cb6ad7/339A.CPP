#include <bits/stdc++.h>
#include <cstring>
using namespace std;

int main() {
  char arr[105];
  char arr3[105];
  int j = 0, k = 0;
  cin.get(arr, 100);

  for (int i = 0; i < strlen(arr); i += 2) {
    for (int j = i + 2; j < strlen(arr); j += 2) {
      if (arr[i] > arr[j])
        swap(arr[i], arr[j]);
    }
  }

  cout << arr;
}
