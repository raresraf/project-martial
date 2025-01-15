#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;

int main() {
  string word;
  int n, i;

  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> word;
    int index = word.length();
    if (index > 10) {
      cout << word[0] << index - 2 << word[index - 1] << endl;
    } else {
      cout << word << endl;
    }
  }
}
