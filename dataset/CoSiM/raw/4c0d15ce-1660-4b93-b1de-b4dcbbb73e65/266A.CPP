#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <map>
#include <stdio.h>
#include <string>

using namespace std;

int main() {
  int n;
  cin >> n;
  string line;
  cin >> line;
  int nr_removed = 0;
  int current_letter = line[0];
  for (auto i = 1; i < n; i++) {
    // cout << line[i] << " " << current_letter << endl;
    if (line[i] == current_letter)
      nr_removed++;
    else
      current_letter = line[i];
  }
  cout << nr_removed;
  return 0;
}