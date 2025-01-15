

#include <bits/stdc++.h>

using namespace std;

int main() {
  string str;
  int c = 0;
  getline(cin, str);
  int n = str.length();
  char char_array[(n + 1)];
  int arr[(n + 1) / 2];
  strcpy(char_array, str.c_str());

  for (int i = 0; i < n + 1; i++) {
    if (i % 2 == 0) {
      arr[c] = (char_array[i]) - 48;
      c++;
    }
  }
  sort(arr, arr + (n + 1) / 2);
  for (int i = 0; i < (n + 1) / 2; i++) {
    if (i != ((n + 1) / 2) - 1) {
      cout << arr[i];
      cout << "+";
    } else if (i == ((n + 1) / 2) - 1) {
      cout << arr[i];
    }
  }

  return 0;
}

// Write your code here
