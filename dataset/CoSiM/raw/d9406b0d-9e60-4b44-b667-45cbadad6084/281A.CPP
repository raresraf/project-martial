

#include <bits/stdc++.h>

using namespace std;

int main() {
  string str;
  getline(cin, str);
  long long n = str.length();
  char char_array[(n + 1)];
  strcpy(char_array, str.c_str());
  char_array[0] = (char)(char_array[0] >= 'a' && char_array[0] <= 'z'
                             ? ((int)char_array[0] - 32)
                             : (int)char_array[0]);
  for (int i = 0; i < n; i++) {
    cout << char_array[i];
  }

  return 0;
}

// Write your code here
