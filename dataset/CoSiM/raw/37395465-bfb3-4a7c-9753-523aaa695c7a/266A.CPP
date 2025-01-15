#include <cstring>
#include <iostream>
#include <stdio.h>
using namespace std;
int main() {
  int n, pairs = 0;
  cin >> n;
  char word[n + 1];
  scanf("%s", word);
  for (int i = 0; i < n - 1; i++) {
    if (word[i] == word[i + 1])
      pairs += 1;
  }
  cout << pairs;
  return 0;
}
