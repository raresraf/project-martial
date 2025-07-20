#include <bits/stdc++.h>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
int main() {
  vector<string> s; // Declaring a vector of strings to store the input
  string p;
  int i = 0, n;
  scanf("%d", &n);
  while (i != n + 1) {
    getline(cin, p);
    s.push_back(p); // reading the string and pushing it to the vector
    i++;
  }

  for (string x : s) {

    auto i = x.begin();    // returns address of the first element to iterator
    auto i2 = x.end() - 1; // returns address of the second elemnt to iterator
                           // i2

    cout << *i;
    cout << x.length() - 2; // There seems to be some error in this part
    cout << *i2 << "\n";
    std::cout << "\n";
    cout << *i2;
  }

} // Here's the link:
  // https://codeforces.com/problemset/problem/71/A
