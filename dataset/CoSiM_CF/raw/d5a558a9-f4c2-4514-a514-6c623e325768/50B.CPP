#include <iostream>
#include <string>
using namespace std;

int main() {
  string str;
  cin >> str;
  int n = str.length(), counter = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (str[i] == str[j] && str[i] != NULL && str[j] != NULL)
        counter = counter + 1;
    }
  }
  cout << counter << endl;
  return 0;
}