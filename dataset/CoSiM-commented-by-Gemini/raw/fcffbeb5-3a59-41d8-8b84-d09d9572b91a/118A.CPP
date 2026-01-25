#include <iostream>
#include <string.h>
using namespace std;

int main() {
  char arr[101];
  int i, j;
  cin.getline(arr, 101);
  int len = strlen(arr);
  for (i = 0; i < len; i++) {
    if (arr[i] >= 65 && arr[i] <= 90) {
      arr[i] = 'a' + (arr[i] - 'A');
    }
    if (arr[i] != 'a' && arr[i] != 'e' && arr[i] != 'i' && arr[i] != 'o' &&
        arr[i] != 'u' && arr[i] != 'y') {
      cout << "." << arr[i];
    }
  }
  cout << endl;
}
