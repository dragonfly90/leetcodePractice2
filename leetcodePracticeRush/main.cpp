//
//  main.cpp
//  leetcodePracticeRush
//
//  Created by Liang Dong on 7/26/16.
//  Copyright © 2016 Liang Dong. All rights reserved.
//

#include<iostream>
#include<vector>
#include<set>
#include<unordered_map>
#include<unordered_set>
#include<string>
#include<queue>



using namespace std;

class SolutionmaxSumSubmatrix {
    
    int maxs;
    
public:
    int maxSumSubmatrix(vector<vector<int>>& matrix, int k) {
        /*
        int m=(int)matrix.size();
        if(m<=0)
            return 0;
        int n=(int)matrix[0].size();
        if(n<=0)
            return 0;
        
        maxs=INT_MIN;
        
        vector<vector<int> > allmatrixsum(m,vector<int>(n,0));
        int area;
        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++){
                area=matrix[i][j];
                if(i-1>=0)
                    area+=allmatrixsum[i-1][j];
                if(j-1>=0)
                    area+=allmatrixsum[i][j-1];
                if(i-1>=0&&j-1>=0)
                    area-=allmatrixsum[i-1][j-1];
                
                allmatrixsum[i][j]=area;
            }
        
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++)
            {
                for(int f=i;f<m;f++)
                {
                    for(int l=j;l<n;l++)
                    {
                        area=allmatrixsum[f][l];
                        if(i-1>=0)
                            area-=allmatrixsum[i-1][l];
                        if(j-1>=0)
                            area-=allmatrixsum[f][j-1];
                        if(i-1>=0&&j-1>=0)
                            area+=allmatrixsum[i-1][j-1];
                        if(area<=k)
                            maxs=max(maxs,area);
                    }
                    
                }
                
            }
        }
        
        return maxs;
         */
        if (matrix.empty()) return 0;
        int row = (int)matrix.size();
        int col =(int) matrix[0].size();
        int res = INT_MIN;
        for (int i = 0; i < col; i++) {
            vector<int> sums(row, 0);
            for (int r = i; r < col; r++) {
                for (int k = 0; k < row; k++) {
                    sums[k] += matrix[k][r];
                }
                
                
                set<int> accuSet;
                accuSet.insert(0);
                int curSum = 0, curMax = INT_MIN;
                for (int sum : sums) {
                    curSum += sum;
                    set<int>::iterator it = accuSet.lower_bound(curSum - k);
                    if (it != accuSet.end()) curMax = std::max(curMax, curSum - *it);
                    accuSet.insert(curSum);
                }
                res = std::max(res, curMax);
            }
        }
        return res;
    }
    
};

int best_cumulative_sum(vector<int> ar,int N,int K)
{
    set<int> cumset;
    cumset.insert(0);
    int best=0,cum=0;
    for(int i=0;i<N;i++)
    {
        cum+=ar[i];
        set<int>::iterator sit=cumset.upper_bound(cum-K);
        
        if(sit!=cumset.end())
            best=max(best,cum-*sit);
        
        cumset.insert(cum);
    }
    return best;
}

class SolutionminSubArrayLen {
    public:
        int minSubArrayLen(int s, vector<int>& nums) {
            /*
            sort(nums.begin(),nums.end(),greater<int>());
            int sum=0;
            vector<int> current;
            for(int i=0;i<nums.size();i++)
            {
                sum+=nums[i];
                if(sum>s)
                    return i+1;
                
            }
            
            return 0;
             */
            if(nums.size()<=0)
                return 0;
            int i=0,j=0,sum=0,minv=INT_MAX;
            while(j<nums.size()){
                sum+=nums[j++];
                while(sum>=s){
                    minv=min(minv,j-i);
                    sum-=nums[i++];
                }
            }
            
            return minv==INT_MAX?0:minv;
        }
};

class SolutioncountRangeSum {
public:
    int countRangeSum(vector<int>& nums, int lower, int upper) {
        int allcounts=0;
        int nsize=(int)nums.size();
        vector<vector<int> > suma(nsize,vector<int>(nsize,0));
        for(int i=0;i<nsize;i++)
            for(int j=i;j<nsize;j++)
            {
                if(j==i)
                    suma[i][j]=nums[i];
                else
                    suma[i][j]=nums[j]+suma[i][j-1];
                
                if(suma[i][j]>=lower&&suma[i][j]<=upper)
                    allcounts++;
            }
        
        return allcounts;
    }
};

//Definition for singly-linked list.
struct ListNode {
        int val;
        ListNode *next;
        ListNode(int x) : val(x), next(NULL) {}
};


class SolutiongetRandom {
    ListNode* currentHead;
    long long length;
public:
    /** @param head The linked list's head.
     Note that the head is guaranteed to be not null, so it contains at least one node. */
    SolutiongetRandom(ListNode* head) {
        length=0;
        currentHead=head;
        ListNode * temp=head;
        while(temp!=NULL)
        {
            length+=1;
            temp=temp->next;
        }
        
    }
    
    /** Returns a random node's value. */
    int getRandom() {
        ListNode * temp=currentHead;
        
        long long currenl=length;
        while(temp!=NULL)
        {
            if((double)rand()/(RAND_MAX)<1/double(currenl))
                return temp->val;
            else
                currenl-=1;
            temp=temp->next;
        }
        return 0;
        
    }
};


class SolutionShuffle{
    vector<int> shuffleVector;
    int nSize;
public:
    SolutionShuffle(vector<int> nums) {
        shuffleVector.assign(nums.begin(),nums.end());
        nSize=(int)shuffleVector.size();
    }
    
    /** Resets the array to its original configuration and return it. */
    vector<int> reset() {
        return shuffleVector;
    }
    
    /** Returns a random shuffling of the array. */
    vector<int> shuffle() {
        vector<int> returnvector;
        returnvector.assign(shuffleVector.begin(),shuffleVector.end());
        for(int j = 1; j < nSize; j++) {
            int i = rand()%(j+1);
            int temp=returnvector[i];
            returnvector[i]=returnvector[j];
            returnvector[j]=temp;
        }
        return returnvector;
        
    }
    
};

class SolutionremoveInvalidParentheses {
    
public:
    vector<string> removeInvalidParentheses(string s) {
        vector<string> allstrs;
        set<string> newallstrs;
        vector<string> tempstrs;
        set<string> strset;
        allstrs.clear();
        allstrs.push_back(s);
        
        while(1)
        {
            strset.clear();
            bool valid=true;
            for(auto tempstr:allstrs)
            {
                if(isvalid(tempstr))
                {
                    strset.insert(tempstr);
                }
                else
                    valid=false;
            }
            
            
            if(!strset.empty())
            {
                return vector<string>(strset.begin(),strset.end());
            }
            
            newallstrs.clear();
            for(auto tempstr:allstrs)
            {
                for(int i=1;i<tempstr.size();i++)
                    newallstrs.insert(tempstr.substr(0,i)+tempstr.substr(i+1));
                newallstrs.insert(tempstr.substr(1));
            }
            //for(auto i:newallstrs)
            //    cout<<i<<endl;
            allstrs.clear();
            allstrs.assign(newallstrs.begin(),newallstrs.end());
            
        }
    }
    
    bool isvalid(string s){
        int ctr=0;
        for(auto i:s)
        {
            if(i=='(')
                ctr+=1;
            else
                if(i==')')
                    ctr-=1;
            if(ctr<0)
                return false;
        }
        
        return ctr==0;
    }
    

    
};

class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        unordered_map<int,int> allset;
        
        for(int i=0;i<numbers.size();i++)
        {
            
            if(allset.find(numbers[i])!=allset.end())
            {
                vector<int> returnIndex;
                returnIndex.push_back(allset[numbers[i]]);
                returnIndex.push_back(i);
                return returnIndex;
            }
            else
                allset[target-numbers[i]]=i;
        }
        return vector<int>();
    }
    
    vector<int> twoSum2(vector<int>& numbers, int target){
        int lo=0, hi=(int)numbers.size()-1;
        while (numbers[lo]+numbers[hi]!=target){
            if (numbers[lo]+numbers[hi]<target){
                lo++;
            } else {
                hi--;
            }
        }
        return vector<int>({lo+1,hi+1});
    }
};

class SolutioncanConstruct {
public:
    bool canConstruct(string ransomNote, string magazine) {
        int magazine_nums[26];
        for(int i=0;i<26;i++)
        {
            magazine_nums[i]=0;
        }
        
        for(auto cm:magazine)
        {
            magazine_nums[int(cm-'a')]++;
        }
        
        for(auto cm:ransomNote)
        {
            magazine_nums[int(cm-'a')]-=1;
            if(magazine_nums[int(cm-'a')]<0)
                return false;
            
        }
       
        return true;
    }
};


class SolutionisSubsequence {
public:
    bool isSubsequence(string s, string t) {
        
        int i=0;
        int j=0;
        while(i<s.size()&&j<t.size())
        {
            if(s[i]==t[j])
            {
                i++;
                j++;
            }
            else
                j++;
        }
        
        if(i==s.size())
            return true;
        else
            return false;
    }
};

// Definition for a binary tree node.
 struct TreeNode {
      int val;
      TreeNode *left;
      TreeNode *right;
      TreeNode(int x) : val(x), left(NULL), right(NULL) {}
  };

class SolutionsumOfLeftLeaves {
public:
    int sumOfLeftLeaves(TreeNode* root) {
        int sum = 0;
        if(!root) return sum;
        preOrder(root,sum);
        return sum;
        
    }
    void preOrder(TreeNode* root,int& sum)
    {
        if(root) {
            if(root->left != NULL && root->left->left == NULL && root->left->right == NULL)
                sum += root->left->val;
            preOrder(root->left, sum);
            preOrder(root->right, sum);
        }
    }
};


class SolutionreconstructQueue {
public:
    vector<pair<int, int>> reconstructQueue(vector<pair<int, int>>& people) {
        auto compa = [](const pair<int, int>& p1, const pair<int, int>& p2)
        { return p1.first > p2.first || (p1.first == p2.first && p1.second < p2.second); };
        sort(people.begin(), people.end(), compa);
        vector<pair<int, int>> res1;
        for (auto& p : people)
            res1.insert(res1.begin() + p.second, p);
        return res1;
    }
};

class SolutionaddStrings {
public:
    string addStrings(string num1, string num2) {
        int i;
        std::reverse(num1.begin(), num1.end());
        std::reverse(num2.begin(), num2.end());
        int len1 = (int)num1.size();
        int len2 = (int)num2.size();
        int minlen;
        
        if(len1>len2)
            minlen = len2;
        else
            minlen = len1;
        
        string sumstr;
        int temp=0;
        int add=0;
      
        for(i=0;i<minlen;i++)
        {
            temp=int(num1[i]-'0')+int(num2[i]-'0')+add;
            cout<<temp<<endl;
            if(temp>=10)
            {
                temp=temp-10;
                add=1;
            }
            else
                add=0;
            sumstr.push_back(char(temp+'0'));
        }
        while(len1>i)
        {
            temp=int(num1[i]-'0')+add;
            if(temp>=10)
            {
                temp=temp-10;
                add=1;
            }
            else
                add=0;
            sumstr.push_back(char(temp+'0'));
            i++;
        }
        while(len2>i)
        {
            temp=int(num2[i]-'0')+add;
            if(temp>=10)
            {
                temp=temp-10;
                add=1;
            }
            else
                add=0;
            sumstr.push_back(char(temp+'0'));
            i++;
        }
        
        if(add!=0)
            sumstr.push_back(char(add+'0'));
        
        std::reverse(sumstr.begin(),sumstr.end());
        return sumstr;
    }
};

/*
 def readBinaryWatch(self, num):
 return ['%d:%02d' % (h, m)
 for h in range(12) for m in range(60)
 if (bin(h) + bin(m)).count('1') == num]
 */


class SolutionreadBinaryWatch {
public:
    string changetotime(int i, int j){
        string str;
        if(j<10)
            str = to_string(i)+":0"+to_string(j);
        else
            str = to_string(i)+":"+to_string(j);
        return str;
    }
    
    int numofdigits(int m)
    {
        int sumd=0;
        while(m>0)
        {
            sumd=sumd+(m&1);
            m=m>>1;
        }
        return sumd;
    }
    vector<string> readBinaryWatch(int num) {
        vector<string> allstrs;
        
        for(int i=0;i<12;i++)
            for(int j=0;j<60;j++)
            {
                if(numofdigits(i)+numofdigits(j)==num)
                {
                    allstrs.push_back(changetotime(i, j));
                }
            }
        
        return allstrs;
    }
};

class SolutiontoHex {
public:
    string toHex(int num) {
        unsigned long positive;
        unsigned long large;
        large=1;
        large=large<<32;
        if(num<0)
        {
            positive=large+num;
        }
        else
            positive=num;
        
        cout<<positive<<endl;
        
        if(positive==0)
            return "0";
        
        string allstrs;
        int temp;
        
        while(positive>0)
        {
            temp=positive%16;
            positive/=16;
            
            if(temp<10)
                allstrs.push_back(char('0'+temp));
            else
            {
                temp-=10;
                allstrs.push_back(char('a'+temp));
            }
        }
        
        reverse(allstrs.begin(), allstrs.end());
        return allstrs;
        
    }
};

class SolutionkthSmallest {
public:
    int kthSmallest(vector<vector<int>>& matrix, int k) {
        int n = (int)matrix.size();
        int le = matrix[0][0], ri = matrix[n-1][n-1];
        int mid = 0;
        while(le < ri)
        {
            mid = le + (ri-le)/2;
            int num = 0;
            for(int i=0; i<n; i++)
            {
                int pos = (int)(upper_bound(matrix[i].begin(), matrix[i].end(), mid) - matrix[i].begin());
                num +=pos;
            }
            if(num<k)
            {
                le = mid +1;
            }
            else
            {
                ri = mid;
            }
        }
        
        return le;
    }
};

class SolutionkthSmallest2 {
public:
    struct compare
    {
        bool operator()(const pair<int,pair<int, int> >& a, const pair<int,pair<int, int> >& b)
        {
            return a.first>b.first;
        }
    };
    int kthSmallest(vector<vector<int>>& arr, int k) {
        
        int n=(int)arr.size();
        int m=(int)arr[0].size();
        
        std::priority_queue< pair<int,pair<int, int> >, vector<pair<int, pair<int, int> > >, compare > p;
        
        for(int i=0;i<n;i++)
            p.push(make_pair(arr[i][0],make_pair(i,0)));
        
        int x=k;
        int ans=k;
        while(x--)
        {
            int e=p.top().first;
            int i=p.top().second.first;
            int j=p.top().second.second;
            ans=e;
            p.pop();
            if(j!=m-1)
            p.push(make_pair(arr[i][j+1],make_pair(i,j+1)));
        }
        return ans;
        
    }
};

template<typename T> void print_queue(T& q) {
    while(!q.empty()) {
        std::cout << q.top() << " ";
        q.pop();
    }
    std::cout << '\n';
}

class SolutionfizzBuzz {
public:
    vector<string> fizzBuzz(int n) {
        vector<string> alls;
        for(int i=1;i<=n;i++)
        {
            if(i%15==0)
            {
                alls.push_back("FizzBuzz");
            }
            else
            {
                if(i%5==0)
                    alls.push_back("Buzz");
                else
                {
                    if(i%3==0)
                        alls.push_back("Fizz");
                    else
                    {
                        
                        alls.push_back(std::to_string(i));
                    }
                }
            }
        }
        
        return alls;
    }
};


class SolutionnumberOfArithmeticSlices {
public:
    int numberOfArithmeticSlices(vector<int>& A) {
        int n=(int)A.size();
        vector<int> dp(n,0);
        
        if(n<3) return 0;
        
        if(A[2]-A[1]==A[1]-A[0])
            dp[2]=1;
        int result = dp[2];
        for(int i=3;i<n;i++){
            if(A[i]-A[i-1]==A[i-1]-A[i-2])
                dp[i]=dp[i-1]+1;
            result+=dp[i];
        }
        return result;
    }
};

class SolutioncountBattleships {
public:
    int countBattleships(vector<vector<char>>& board) {
        int count = 0;
        for(int i=0;i<board.size();i++)
        for(int j=0;j<board[0].size();j++)
        if(board[i][j]=='X' && (i==0 || board[i-1][j]!='X') && (j==0 || board[i][j-1]!='X')) count++;
        return count;
    }
};

class SolutionhammingDistance {
public:
    int hammingDistance(int x, int y) {
        int z=x^y;
        int sumn=0;
        while(z!=0)
        {
            sumn+=z&1;
            z=z>>1;
        }
        return sumn;
    }
};

/*
 public class Solution {
	Map<String,List<String>> map;
	List<List<String>> results;
 public List<List<String>> findLadders(String start, String end, Set<String> dict) {
 results= new ArrayList<List<String>>();
 if (dict.size() == 0)
 return results;
 
 int min=Integer.MAX_VALUE;
 
 Queue<String> queue= new ArrayDeque<String>();
 queue.add(start);
 
 map = new HashMap<String,List<String>>();
 
 Map<String,Integer> ladder = new HashMap<String,Integer>();
 for (String string:dict)
 ladder.put(string, Integer.MAX_VALUE);
 ladder.put(start, 0);
 
 dict.add(end);
 //BFS: Dijisktra search
 while (!queue.isEmpty()) {
 
 String word = queue.poll();
 
 int step = ladder.get(word)+1;//'step' indicates how many steps are needed to travel to one word.
 
 if (step>min) break;
 
 for (int i = 0; i < word.length(); i++){
 StringBuilder builder = new StringBuilder(word);
 for (char ch='a';  ch <= 'z'; ch++){
 builder.setCharAt(i,ch);
 String new_word=builder.toString();
 if (ladder.containsKey(new_word)) {
 
 if (step>ladder.get(new_word))//Check if it is the shortest path to one word.
 continue;
 else if (step<ladder.get(new_word)){
 queue.add(new_word);
 ladder.put(new_word, step);
 }else;// It is a KEY line. If one word already appeared in one ladder,
 // Do not insert the same word inside the queue twice. Otherwise it gets TLE.
 
 if (map.containsKey(new_word)) //Build adjacent Graph
 map.get(new_word).add(word);
 else{
 List<String> list= new LinkedList<String>();
 list.add(word);
 map.put(new_word,list);
 //It is possible to write three lines in one:
 //map.put(new_word,new LinkedList<String>(Arrays.asList(new String[]{word})));
 //Which one is better?
 }
 
 if (new_word.equals(end))
 min=step;
 
 }//End if dict contains new_word
 }//End:Iteration from 'a' to 'z'
 }//End:Iteration from the first to the last
 }//End While
 
 //BackTracking
 LinkedList<String> result = new LinkedList<String>();
 backTrace(end,start,result);
 
 return results;
 }
 private void backTrace(String word,String start,List<String> list){
 if (word.equals(start)){
 list.add(0,start);
 results.add(new ArrayList<String>(list));
 list.remove(0);
 return;
 }
 list.add(0,word);
 if (map.get(word)!=null)
 for (String s:map.get(word))
 backTrace(s,start,list);
 list.remove(0);
 }
 }
 */

class SolutionfindLadders {
public:
    vector<vector<string>> findLadders(string beginWord, string endWord, unordered_set<string> &dict) {
        vector<vector<string> > paths;
        vector<string> path(1, beginWord);
        if(beginWord==endWord)
        {
            paths.push_back(path);
            return paths;
        }
        unordered_set<string> forward, backward;
        forward.insert(beginWord);
        backward.insert(endWord);
        unordered_map<string, vector<string> > tree;
        bool reversed = false;
        if(buildTree(forward,backward,dict,tree,reversed))
            getPath(beginWord, endWord, tree, path, paths);
        return paths;
    }
private:
    bool buildTree(unordered_set<string> &forward, unordered_set<string> &backward, unordered_set<string> &dict, unordered_map<string, vector<string> > &tree, bool reversed)
    {
        if (forward.empty()) return false;
        if (forward.size() > backward.size())
        return buildTree(backward, forward, dict, tree, !reversed);
        for (auto &word: forward) dict.erase(word);
        for (auto &word: backward) dict.erase(word);
        unordered_set<string> nextLevel;
        bool done = false; //in case of invalid further searching;
        for (auto &it: forward) //traverse each word in the forward -> the current level of the tree;
        {
            string word = it;
            for (auto &c: word)
            {
                char c0 = c; //store the original;
                for (c = 'a'; c <= 'z'; ++c) //try each case;
                {
                    if (c != c0) //avoid futile checking;
                    {
                        if (backward.count(word))  //using count is an accelerating method;
                        {
                            done = true;
                            !reversed ? tree[it].push_back(word) : tree[word].push_back(it); //keep the tree generation direction consistent;
                        }
                        else if (!done && dict.count(word))
                        {
                            nextLevel.insert(word);
                            !reversed ? tree[it].push_back(word) : tree[word].push_back(it);
                        }
                    }
                }
                c = c0; //restore the word;
            }
        }
        return done || buildTree(nextLevel, backward, dict, tree, reversed);
    }
    
    void getPath(string &beginWord, string &endWord, unordered_map<string, vector<string> > &tree, vector<string> &path, vector<vector<string> > &paths) //using reference can accelerate;
    {
        if (beginWord == endWord) paths.push_back(path); //till the end;
        else
        {
            for (auto &it: tree[beginWord])
            {
                path.push_back(it);
                getPath(it, endWord, tree, path, paths); //DFS retrieving the path;
                path.pop_back();
            }
        }
    }
};

class SolutionminMoves2 {
public:
    int minMoves2(vector<int>& nums) {
        sort(nums.begin(),nums.end());
        //Arrays.sort(nums);
        int i = 0, j = (int)nums.size()-1;
        int count = 0;
        while(i < j){
            count += nums[j]-nums[i];
            i++;
            j--;
        }
        return count;
    }
};

/*
 Arrays.sort(g);
 Arrays.sort(s);
 int i = 0;
 for(int j=0;i<g.length && j<s.length;j++) {
	if(g[i]<=s[j]) i++;
 }
 return i;
 */

class SolutionfindContentChildren {
public:
    int findContentChildren(vector<int>& g, vector<int>& s) {
        sort(g.begin(),g.end());
        sort(s.begin(),s.end());
        int i = 0;
        for(int j=0;i<g.size()&&j<s.size();j++){
            if(g[i]<=s[j])
                i++;
        }
        
        return i;
                    
    }
};


class SolutionfindDisappearedNumbers {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        vector<int> ret;
        for(int i=0;i<nums.size();i++){
            int val = abs(nums[i])-1;
            if(nums[val]>0){
                nums[val] = -nums[val];
            }
        }
        
        for(int i=0;i<nums.size();i++){
            if(nums[i]>0){
                ret.push_back(i+1);
            }
        }
        
        return ret;
    }
};

class SolutiontotalHammingDistance {
public:
    int countn(int a){
        int sumn=0;
        while(a>0)
        {
            sumn+=a&1;
            a>>=1;
        }
        return sumn;
    }
    int totalHammingDistance(vector<int>& nums) {
        int total = 0, n = (int)nums.size();
        for (int j=0;j<32;j++) {
            int bitCount = 0;
            for (int i=0;i<n;i++)
            bitCount += (nums[i] >> j) & 1;
            total += bitCount*(n - bitCount);
        }
        return total;
    }
};


class SolutionminMoves {
public:
    unsigned long sumn(vector<int>& nums){
        unsigned long sumn=0;
        for(auto i:nums)
        {
            sumn+=i;
        }
        return sumn;
    }
    
    int minMoves(vector<int>& nums) {
        return (int)(sumn(nums) - nums.size() * *min_element(begin(nums), end(nums)));
    }
};

int numberOfBoomerangs(vector<pair<int, int>>& points) {
    int booms = 0;
    for (auto &p : points) {
        unordered_map<double, int> ctr(points.size());
        for (auto &q : points)
        booms += 2 * ctr[hypot(p.first - q.first, p.second - q.second)]++;
    }
    return booms;
}

class SolutionlongestPalindrome {
// Lowercase letters a-z ASCII 97 to 122
// Uppercase letters A-Z ASCII 65-90
public:
    int longestPalindrome(string s) {
        int lowercase[36]={0};
        int upercase[36]={0};
        
        for(int i=0;i<s.size();i++)
        {
            if(s[i]>=97&&s[i]<=122)
            lowercase[s[i]-'a']++;
            
            if(s[i]>=65&&s[i]<=90)
            upercase[s[i]-'A']++;
        }
        
        int length=0;
        for(int i=0;i<36;i++)
        {
            lowercase[i]/=2;
            length+=lowercase[i]*2;
        }
        
        for(int i=0;i<36;i++)
        {
            upercase[i]/=2;
            length+=upercase[i]*2;
        }
        
        bool single=false;
        for(int i=0;i<36;i++)
        {
            if(lowercase[i]==0)
            {
                single = true;
                break;
            }
            
            if(upercase[i]==0)
            {
                single = true;
                break;
            }
            
        }
        
        if(single)
            return length;
        else
            return length+1;
    }
};

class SolutionlongestPalindrome2 {
    // Lowercase letters a-z ASCII 97 to 122
    // Uppercase letters A-Z ASCII 65-90
public:
    int longestPalindrome(string s) {
        if(s.size()==0)
            return 0;
        unordered_set<char> hs;
        int count=0;
        for(int i=0;i<s.size();i++){
            if(hs.find(s[i])!=hs.end())
            {
                hs.erase(s[i]);
                count++;
            }
            else{
                hs.insert(s[i]);
            }
        }
        
        if(!hs.empty())
        return count*2+1;
        
        return count*2;
    }
};

/*
 public int longestPalindrome(String s) {
 if(s==null || s.length()==0) return 0;
 HashSet<Character> hs = new HashSet<Character>();
 int count = 0;
 for(int i=0; i<s.length(); i++){
 if(hs.contains(s.charAt(i))){
 hs.remove(s.charAt(i));
 count++;
 }else{
 hs.add(s.charAt(i));
 }
 }
 if(!hs.isEmpty()) return count*2+1;
 return count*2;
 }
 */

class SolutionlicenseKeyFormatting {
public:
    string licenseKeyFormatting(string S, int K) {
        string ret;
        
        int nstr = 0;
        for (auto i:S)
            nstr += (S[i] != '-');
        
        int u = 0;
        for (int i = 0; i < nstr; ++ i){
            while (S[u] == '-')
             u++;
            ret += toupper(S[u]);
            if (((nstr - i - 1) % K == 0) && (i != nstr - 1)) ret += '-';
            u++;
        }
        return ret;
        /*
        string res;
        for (auto i = S.rbegin(); i < S.rend(); i++)
            if (*i != '-')
            (res.size()%(K+1)-K? res : res+='-') += toupper(*i);
        return reverse(res.begin(), res.end()), res;
         */
    }
};


class SolutionfindMaxConsecutiveOnes {
public:
    int findMaxConsecutiveOnes(vector<int>& nums) {
        int length=0;
        int maxlength=0;
        for(auto i:nums)
        {
            if(i==1)
            {
                length++;
                if(length>maxlength)
                    maxlength=length;
            }
            else
                length=0;
        }
        return maxlength;
    }
};

class SolutionfrequencySort {
public:
    string frequencySort(string s) {
        unordered_map<char,int> count;
        vector<string> bucket(s.size()+1, "");
        string res;
        
        
        for(char c:s)
            count[c]++;
        
        for(auto& it:count) {
            int n = it.second;
            char c = it.first;
            bucket[n].append(n, c);
        }
        //form descending sorted string
        for(int i=(int)s.size(); i>0; i--) {
            if(!bucket[i].empty())
            res.append(bucket[i]);
        }
        return res;
    }
};

class SolutionfindComplement {
public:
    int findComplement(int num) {
        int sizek=0;
        int currentnum=num;
        while(currentnum>0)
        {
            currentnum>>=1;
            sizek++;
        }
        
        return (1<<sizek)-num-1;
        
    }
};


class SolutionfindDuplicates {
public:
    vector<int> findDuplicates(vector<int>& nums) {
        vector<int> duplicates;
        for (int i = 0; i < nums.size(); ++i) {
            int index = nums[i] & 0x7fffffff;
            if (nums[index - 1] < 0) {
                duplicates.push_back(index);
            }
            else {
                nums[index - 1] *=-1;
            }
        }
        return duplicates;
        
    }
};

class SolutionfindContentChildrenc {
public:
    int findContentChildren(vector<int>& g, vector<int>& s) {
        sort(g.begin(),g.end());
        sort(s.begin(),s.end());
        int i,j;
        
        for(i=0,j=0;i<g.size()&&j<s.size();)
        {
            if(s[j]>=g[i])
            {
                i++;
                j++;
            }
            else{
                j++;
            }
        }
        
        return i;
    }
};

class SolutionfindWords {
public:
    vector<string> findWords(vector<string>& words) {
        char a[]={'q','w','e','r','t','y','u','i','o','p'};
        char b[]={'a','s','d','f','g','h','j','k','l'};
        char c[]={'z','x','c','v','b','n','m'};
        unordered_map<char, int> mapschar;
        vector<string> allstrs;
        
        for(auto ch:a)
        {
            mapschar[ch]=1;
        }
        for(auto ch:b)
        {
            mapschar[ch]=2;
        }
        for(auto ch:c)
        {
            mapschar[ch]=3;
        }
        
        int origin=-1;
        bool findT;
        for(int i=0; i<words.size();i++)
        {
            findT=true;
            origin=-1;
            for(auto j:words[i])
            {
                if(origin==-1)
                    origin=mapschar[j];
                
                if(origin!=mapschar[j])
                {
                    findT=false;
                    break;
                    
                }
            }
            if(findT==true)
                allstrs.push_back(words[i]);
            origin=1;
            
        }
        
        return allstrs;
    }
};

/*
 std::unordered_set <char> dict1 = { 'q','Q','w','W','e','E','r','R','t','T','y','Y','u','U','i','I','o','O','p','P' };
 std::unordered_set <char> dict2 = { 'a','A','s','S','d','D','f','F','g','G','h','H','j','J','k','K','l','L'};
 std::unordered_set <char> dict3 = { 'z','Z','x','X','c','C','v','V','b','B','n','N','m','M'};
 
 vector<string> res;
 
 for (auto &word : words)
 {
 bool d1 = true, d2 = true, d3 = true;
 
 for (auto & ch : word)
 {
 if (d1) {
 auto it = dict1.find(ch);
 if (it == dict1.end()) d1 = false;
 }
 
 if (d2) {
 auto it = dict2.find(ch);
 if (it == dict2.end()) d2 = false;
 }
 
 if (d3) {
 auto it = dict3.find(ch);
 if (it == dict3.end()) d3 = false;
 }
 }
 
 if (d1 || d2 || d3) res.push_back(word);
 }
 
 return res;
*/

class SolutionlargestValues {
    
    queue<TreeNode* > solutions;
    queue<int> levels;
    
    TreeNode* current;
    int currentlevel;
    vector<int> smallest;
    
public:
    vector<int> largestValues(TreeNode* root) {
        if(root==NULL)
            return vector<int>();
        
        levels.push(0);
        solutions.push(root);
        
        while(!solutions.empty())
        {
            current = solutions.front();
            currentlevel = levels.front();
            
            solutions.pop();
            levels.pop();
            
            if(currentlevel==smallest.size())
            {
                smallest.push_back(current->val);
            }
            else
                if(current->val>smallest[currentlevel])
                {
                    smallest[currentlevel]=current->val;
                }
            
            if(current->left!=NULL)
            {
                solutions.push(current->left);
                levels.push(currentlevel+1);
                
            }
            if(current->right!=NULL)
            {
                solutions.push(current->right);
                levels.push(currentlevel+1);
            }
            
            
        }
        
        return smallest;
    }
};

class SolutionlargestNumber{
public:
    string largestNumber(vector<int>& num){
        /*
        vector<string> arr;
        for(auto i:num)
            arr.push_back(to_string(i));
        sort(begin(arr),end(arr),[](string &s1, string &s2){return s1+s2>s2+s1;});
        string res;
        for(auto s:arr)
            res+=s;
        while(res[0]=='0'&&res.length()>1)
            res.erase(0,1);
        return res;
         */
        
        /*
        sort(num.begin(), num.end(), [](int a, int b){
            return to_string(a)+to_string(b) > to_string(b)+to_string(a);
        });
        string ans;
        for(int i=0; i<num.size(); i++){
            ans += to_string(num[i]);
        }
        return ans[0]=='0' ? "0" : ans;
        */
        
        vector<string> arr;
        for(auto i:num)
            arr.push_back(to_string(i));
        sort(begin(arr),end(arr),[](string &s1, string &s2){return s1+s2>s2+s1;});
        string ans;
        for(int i=0; i<arr.size(); i++){
            ans += arr[i];
        }
        return ans[0]=='0' ? "0" : ans;
    }
};


class SolutioncountArrangement {
public:
    int countArrangement(int N) {
        vector<int> vs;
        for (int i=0; i<N; ++i) vs.push_back(i+1);
        return counts(N, vs);
    }
    int counts(int n, vector<int>& vs) {
        if (n <= 0) return 1;
        int ans = 0;
        for (int i=0; i<n; ++i) {
            if (vs[i]%n==0 || n%vs[i]==0) {
                swap(vs[i], vs[n-1]);
                ans += counts(n-1, vs);
                swap(vs[i], vs[n-1]);
            }
        }
        return ans;
    }
};

class SolutionconvertBST {
public:
    TreeNode* convertBST(TreeNode* root) {
        if(root==NULL)
            return NULL;
        DFS(root,0);
        return root;
        
    }
    
    int DFS(TreeNode* root, int preSum)
    {
       if(root->right!=NULL)
           preSum = DFS(root->right, preSum);
        
        root->val = root->val + preSum;
        
        return (root->left!=NULL)?DFS(root->left,root->val):root->val;
        
        
    }
};

#include<string>
class Solutionencode {
    unordered_map<int, string> allstrs;
    int i;

public:
    Solutionencode()
    {
        i=0;
    }
    // Encodes a URL to a shortened URL.
    string encode(string longUrl) {
        allstrs[i]=longUrl;
        return "http://tinyurl.com/"+to_string(i);
        i++;
    }
    
    // Decodes a shortened URL to its original URL.
    string decode(string shortUrl) {
        shortUrl.erase(0,19);
        int j=stoi(shortUrl);
        return allstrs[j];
        
    }
};


class Solutiondesign {
    
public:
    Solutiondesign(){
        shortToLongUrlMapping.clear();
        srand (time(NULL));
    }
    
    // Encodes a URL to a shortened URL.
    string encode(string longUrl) {
        string shortUrl('a',5);
        int idx = 0, randomVal = 0;
        while(shortToLongUrlMapping.find(shortUrl) != shortToLongUrlMapping.end()){
            randomVal = rand() % 62;
            if(randomVal >= 0 && randomVal < 26)
                shortUrl[idx] = ('A' + randomVal);
            else if(randomVal >= 26 && randomVal < 52)
                shortUrl[idx] = ('a' + (randomVal - 26));
            else
                shortUrl[idx] = ('0' + (randomVal - 52));
            idx = (idx + 1)%5;
        }
        shortToLongUrlMapping[shortUrl] = longUrl;
        return shortUrl;
    }
    
    // Decodes a shortened URL to its original URL.
    string decode(string shortUrl) {
        string longUrl = shortToLongUrlMapping.find(shortUrl) != shortToLongUrlMapping.end() ? shortToLongUrlMapping[shortUrl] : shortUrl;
        return longUrl;
    }
private:
    unordered_map<string,string> shortToLongUrlMapping;
};


class SolutiondetectCapitalUse {
public:
    bool detectCapitalUse(string word) {
        bool upperword=true;
        
        if(word.size()==0)
            return true;
        
        if(isupper(word[0]))
            upperword=true;
        else
            upperword=false;
        
        for(int i=1;i<word.size();i++)
        {
            if(!upperword&&isupper(word[i]))
                return false;
            if(upperword)
            {
                if(i==1)
                {
                    if(!isupper(word[i]))
                        upperword=false;
                }
                else{
                    if(!isupper(word[i]))
                        return false;
                }
            
            }
            
        }
        
        return true;
        
    }
    /*
     
     def detectCapitalUse(self, word):
        return word.isupper() or word.islower() or word.istitle()
     
    */
};

/*
 class Solution(object):
 def arrayPairSum(self, nums):
 nums.sort()
 sumn = 0
 for i in range(0, len(nums), 2):
 sumn = sumn + nums[i]
 
 return sumn
 */

// Your Solution object will be instantiated and called as such:
// Solution solution;
// solution.decode(solution.encode(url));

#include <string>
#include <sstream>
#include <vector>
#include <iterator>

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}


class SolutioncomplexNumberMultiply {
public:
    string complexNumberMultiply(string a, string b) {
        vector<string> elems1,elems2;
        elems1=split(a,'+');
        elems2=split(b,'+');
        
        int elems1_1=stod(elems1[0]);
        int elems1_2=stod(elems1[1].substr(0, elems1[1].size()-1));
        
        int elems2_1=stod(elems2[0]);
        int elems2_2=stod(elems2[1].substr(0, elems2[1].size()-1));

        
        int realpart = elems1_1*elems2_1-elems1_2*elems2_2;
        int pospart = elems1_1*elems2_2+elems1_2*elems2_1;
        
        return to_string(realpart)+'+'+to_string(pospart);
    }
};

bool comp(pair<char,int> a, pair<char,int> b) {
    return a.second < b.second;
}

class SolutionfindFrequentTreeSum {
    unordered_map<int, int> maps;

public:
    int sumTree(TreeNode* croot){
        int sumvalue=0;
        if(croot->left!=NULL)
        {
            sumvalue = sumvalue+sumTree(croot->left);
        }
        if(croot->right!=NULL)
        {
            sumvalue = sumvalue+sumTree(croot->right);
        }
        sumvalue=sumvalue+croot->val;
        if(maps.find(sumvalue)!=maps.end())
            maps[sumvalue]=maps[sumvalue]+1;
        else
            maps[sumvalue]=0;
        return 1;
    }
    
    vector<int> findFrequentTreeSum(TreeNode* root) {
        maps.clear();
        vector<std::pair<int, int>> elems(maps.begin(), maps.end());
        sort(elems.begin(), elems.end(), comp);
        
        vector<int> items;
        for (size_t it = 0; it < elems.size(); ++it) {
            items.push_back(elems[it].first);
        }
        return items;
    }
};

class SolutionfindPoisonedDuration {
public:
    int findPoisonedDuration(vector<int>& timeSeries, int duration) {
        int total_duration=0;
        
        if(timeSeries.size()==0)
            return 0;
        
        if(timeSeries.size()==1)
            return duration;
            
        for(int i=1;i<timeSeries.size();i++)
        {
                if(timeSeries[i]>timeSeries[i-1]+duration)
                {
                    total_duration=total_duration+duration;
                }
                else
                    total_duration=total_duration+timeSeries[i]-timeSeries[i-1];
        }
        
        total_duration=total_duration+duration;
        
        return total_duration;
    }
};

/*
 
 class Solution {
 vector<vector<string> > ans;
 public:
 vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {
 unordered_set<string> wordListset(wordList.begin(), wordList.end());
 wordListset.insert(endWord);
 int dsize = wordListset.size(), len = beginWord.length();
 unordered_map<string, vector<string> > next;
 unordered_map<string, int> vis;
 queue<string> q;
 vector<string> path;
 ans.clear();
 q.push(beginWord);
 vis[beginWord] = 0;
 while (!q.empty()) {
 string s = q.front(); q.pop();
 if (s == endWord) break;
 int step = vis[s];
 vector<string> snext;
 for (int i = 0; i < len; i++) {
 string news = s;
 for (char c = 'a'; c <= 'z'; c++) {
 news[i] = c;
 if (c == s[i] || wordListset.find(news) == wordListset.end()) continue;
 auto it = vis.find(news);
 if (it == vis.end()) {
 q.push(news);
 vis[news] = step + 1;
 }
 snext.push_back(news);
 }
 }
 next[s] = snext;
 }
 path.push_back(beginWord);
 dfspath(path, next, vis, beginWord, endWord);
 return ans;
 }
 
 void dfspath(vector<string> &path,  unordered_map<string, vector<string> > &next,
 unordered_map<string, int> &vis, string now, string end){
 if (now == end) ans.push_back(path);
 else {
 auto vec = next[now];
 int visn = vis[now];
 for (int i = 0; i < vec.size(); i++) {
 if (vis[vec[i]] != visn + 1) continue;
 path.push_back(vec[i]);
 dfspath(path, next, vis, vec[i], end);
 path.pop_back();
 }
 }
 }
 };
 
 */

class SolutionfindRestaurant {
public:
    vector<string> findRestaurant(vector<string>& list1, vector<string>& list2) {
        unordered_map<string,int> firstindex;
        unordered_map<string, int> numbers;
        
        return vector<string>();
    }
};

#include <iostream>
#include <memory>

struct Foo {
    Foo(){ std::cout<<"Foo::Foo\n";}
    ~Foo(){ std::cout<<"Foo::~Foo\n";}
    void bar(){ std::cout<<"Foo::bar\n";}
};

void f(const Foo &foo)
{
    std::cout<<"f(const Foo&)\n";
}


class SolutiondiameterOfBinaryTree {
public:
    int maxdiadepth = 0;
    
    int dfs(TreeNode* root){
        if(root == NULL) return 0;
        
        int leftdepth = dfs(root->left);
        int rightdepth = dfs(root->right);
        
        if(leftdepth + rightdepth > maxdiadepth) maxdiadepth = leftdepth + rightdepth;
        return max(leftdepth +1, rightdepth + 1);
    }
    
    int diameterOfBinaryTree(TreeNode* root) {
        dfs(root);
        
        return maxdiadepth;
    }
};

bool compareString (string i,string j) { return (i.compare(j)<0); }
int comparedifference(string i, string j){
    if(i.size()!=5||j.size()!=5)
        return -1;
    int sumdiff=0;
    sumdiff+=int(j[0]-i[0])*10*60;
    sumdiff+=int(j[1]-i[1])*60;
    sumdiff+=int(j[3]-i[3])*10;
    sumdiff+=int(j[4]-i[4]);
    return sumdiff;
}

class SolutionfindMinDifference {
public:
    int findMinDifference(vector<string>& timePoints) {
        int maxdiff = 60*25;
        int difference;
        sort(timePoints.begin(),timePoints.end(),compareString);
        for(int i = 0; i < timePoints.size()-1; i++){
            difference = comparedifference(timePoints[i], timePoints[i+1]);
            cout<<difference<<endl;
            if(difference < maxdiff)
                maxdiff = difference;
            
        }
        
        difference = comparedifference(timePoints.back(), "23:60")+
                     comparedifference("00:00", timePoints.front());
        
        cout<<difference<<endl;
        if(difference < maxdiff)
            maxdiff = difference;
        return maxdiff;
        
    }
};

class SolutionfindDiagonalOrder {
public:
    vector<int> findDiagonalOrder(vector<vector<int>>& matrix) {
        if(matrix.size()==0)
            return vector<int>();
        int h = (int)matrix.size();
        int w = (int)matrix[0].size();
        int id = 0;
        vector<int> res;
        res.resize(h*w);
        for(int i=0; i<h+w; i++){
            int lb = (int)max(0, i-w+1), ub = (int)min(i,h-1);
            if(i%2 == 0)
                for(int j=ub; j>=lb;j--)
                    res[id++] = matrix[j][i-j];
            else
                for(int j=lb;j<=ub;j++)
                    res[id++] = matrix[j][i-j];
        }
        return res;
        
    }
};

class SolutionfindLadders2 {
    vector<vector<string> > ans;
public:
    vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> wordListset(wordList.begin(), wordList.end());
        wordListset.insert(endWord);
        int dsize = static_cast<int>(wordListset.size());
        int len = static_cast<int>(beginWord.length());
        unordered_map<string, vector<string> > next;
        unordered_map<string, int> vis;
        queue<string> q;
        vector<string> path;
        ans.clear();
        q.push(beginWord);
        vis[beginWord] = 0;
        while (!q.empty()) {
            string s = q.front(); q.pop();
            if (s == endWord) break;
            int step = vis[s];
            vector<string> snext;
            for (int i = 0; i < len; i++) {
                string news = s;
                for (char c = 'a'; c <= 'z'; c++) {
                    news[i] = c;
                    if (c == s[i] || wordListset.find(news) == wordListset.end()) continue;
                    auto it = vis.find(news);
                    if (it == vis.end()) {
                        q.push(news);
                        vis[news] = step + 1;
                    }
                    snext.push_back(news);
                }
            }
            next[s] = snext;
        }
        path.push_back(beginWord);
        dfspath(path, next, vis, beginWord, endWord);
        return ans;
    }
    
    void dfspath(vector<string> &path,  unordered_map<string, vector<string> > &next,
                 unordered_map<string, int> &vis, string now, string end){
        if (now == end) ans.push_back(path);
        else {
            auto vec = next[now];
            int visn = vis[now];
            for (int i = 0; i < vec.size(); i++) {
                if (vis[vec[i]] != visn + 1) continue;
                path.push_back(vec[i]);
                dfspath(path, next, vis, vec[i], end);
                path.pop_back();
            }
        }
    }
};

class Solutiontree2str {
    
public:
    string tree2str(TreeNode* t) {
        if(t==NULL)
            return string();
        if(t->left==NULL&&t->right!=NULL)
            return std::to_string(t->val)+"()"+"("+tree2str(t->right)+")";
        if(t->left!=NULL&&t->right==NULL)
            return std::to_string(t->val)+"("+tree2str(t->left)+")";
        if(t->left!=NULL&&t->right!=NULL)
            return std::to_string(t->val)+"("+tree2str(t->left)+")"+"("+tree2str(t->right)+")";
        if(t->left==NULL&&t->right==NULL)
            return std::to_string(t->val);
        return string();
    }
};

class SolutionfindRelativeRanks {
    
public:
    vector<string> findRelativeRanks(vector<int>& nums) {
        vector<pair<int, int> > allpairs;
        for(int i = 0; i < nums.size(); i++){
            allpairs.push_back(make_pair(i, nums[i]));
            
        }
        sort(allpairs.begin(), allpairs.end(), [](pair<int, int> a,
                                                  pair<int, int> b){
            return a.second>b.second;
        });
        
        vector<string> returnstr;
        returnstr.resize(nums.size());
        for(int i =0; i<allpairs.size(); i++)
        {
            if (i==0)
                returnstr[allpairs[i].first] = "Gold Medal";
            else
                if (i==1)
                    returnstr[allpairs[i].first] = "Silver Medal";
                else
                    if (i==2)
                        returnstr[allpairs[i].first] = "Bronze Medal";
                    else
                        returnstr[allpairs[i].first] = to_string(i+1);
        }
        return returnstr;
    }
};

class SolutionaddOneRow {
    void addRowInDepth(TreeNode* root, const int& v, const int& d, int currentd) {
        if(currentd==d)
        {
            TreeNode* left = root->left;
            TreeNode* right = root->right;
            TreeNode* newleft = new TreeNode(v);
            TreeNode* newright = new TreeNode(v);
            root->left = newleft;
            root->right = newright;
            newleft->left = left;
            newright->right = right;
        }
        else{
            addRowInDepth(root->left, v, d, currentd+1);
            addRowInDepth(root->right, v, d, currentd+1);
        }
    }
public:
    TreeNode* addOneRow(TreeNode* root, int v, int d) {
        if(root==NULL)
            return NULL;
        addRowInDepth(root, v, d, 1);
        return root;
    }
};

#include<stack>

class SolutionnextGreaterElements {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        int n = static_cast<int>(nums.size());
        vector<int> next(n, -1);
        stack<int> s; // index stack
        for (int i = 0; i < n * 2; i++) {
            int num = nums[i % n];
            while (!s.empty() && nums[s.top()] < num) {
                next[s.top()] = num;
                s.pop();
            }
            if (i < n) s.push(i);
        }
        return next;
    }
};

class SolutionmaxCount {
public:
    int maxCount(int m, int n, vector<vector<int>>& ops) {
        int maxM=m;
        int maxN=n;
        for(int i=0; i<ops.size(); i++){
            if(ops[i][0]<maxM)
                maxM=ops[i][0];
            if(ops[i][1]<maxN)
                maxN=ops[i][1];
        }
        return maxM * maxN;
    }
};

class SolutionconstructMaximumBinaryTree2 {
public:
    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        vector<TreeNode* > stk;
        for(int i = 0; i < nums.size(); ++i )
        {
            TreeNode* cur = new TreeNode(nums[i]);
            while(!stk.empty() && stk.back()->val < nums[i])
            {
                cur->left = stk.back();
                stk.pop_back();
            }
            
            if(!stk.empty())
                stk.back()->right = cur;
            
            stk.push_back(cur);
        }
        
        return stk.front();
    }
};

class SolutionconstructMaximumBinaryTree {
public:
    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        return constructMaximumRange(nums, 0, static_cast<int>(nums.size())-1);
    }
    
    TreeNode* constructMaximumRange(vector<int>& nums, int i, int j){
        TreeNode* root=NULL;
        if(i>j)
            return NULL;
        if(i==j)
        {
            root = new TreeNode(nums[0]);
            return root;
        }
        
        int max_num = nums[i];
        int max_index = i;
        vector<int> all_nums;
        for(int k=i+1; k<=j; k++)
        {
            if(nums[k]>max_num)
            {
                max_num=nums[k];
                max_index=k;
            }
        }
        
        root->left = constructMaximumRange(nums, i, max_index);
        root->right = constructMaximumRange(nums, max_index+1, j);
        return root;

    }
};

class SolutionprintTree {
    
    int maxd;
public:
    int depthOfTree(TreeNode* root){
        if(root==NULL)
            return 0;
        int maxdepth = max(depthOfTree(root->left), depthOfTree(root->right)) + 1;
        return maxdepth;
    }
    
    int widthOfTree(int depth){
        return static_cast<int>(pow(2, depth)-1);
    }
    
    void preOrder(vector<vector<string>>& allNumbers, TreeNode* root, int d, int posa){
        allNumbers[maxd-d-1][posa]=std::to_string(root->val);
        if(root->left!=NULL)
            preOrder(allNumbers, root, d-1, posa-pow(2,d-1));
        if(root->right!=NULL)
            preOrder(allNumbers, root, d-1, posa+pow(2,d-1));
    }
                   
    vector<vector<string>> printTree(TreeNode* root) {
        int d = depthOfTree(root);
        maxd = d;
        int width = widthOfTree(d);
        vector<vector<string> > allNumbers(d, vector<string>(width, ""));
        preOrder(allNumbers, root, d-1, pow(2,d-1)-1);
        return allNumbers;
    }
    
};

class SolutionfindTarget {
    unordered_set<int> allset;
    bool visitNode(TreeNode* root, int target) {
        if(root==NULL)
            return false;
        int currentVal = root->val;
        if(allset.find(currentVal)!=allset.end()) {
            return true;
        } else {
            allset.insert(target-currentVal);
        }
        if(root->left!=NULL && visitNode(root->left, target))
            return true;
        if(root->right!=NULL && visitNode(root->right, target))
            return true;
        return false;
    }
public:
    bool findTarget(TreeNode* root, int k) {
        allset.clear();
        return visitNode(root, k);
    }
};

struct Interval {
    int start;
    int end;
    Interval(): start(0), end(0) {}
    Interval(int s, int e): start(s), end(e){}
};

// [1 2] [2 3] [1.5 2.5] [6 7]
class SolutioneraseOverlapIntervals {
public:
    int eraseOverlapIntervals(vector<Interval>& intervals) {
        sort(intervals.begin(), intervals.end(), [](Interval interval1, Interval interval2){
            return interval1.end < interval2.end;
        });
        if(intervals.size()==0)
             return 0;
    
        
        int currentend = intervals[0].end;
        int removeRangeNumber = 0;
        for(int i = 1; i < intervals.size(); i++)
        {
            if(intervals[i].start < currentend)
            {
                removeRangeNumber++;
            }
            else
            {
                currentend = intervals[i].end;
            }
        }
        
        return removeRangeNumber;
    }
};

int main(){
    
    std::string stringvalues = "125-320-52-75-333";
    std::istringstream iss (stringvalues);
    char _;
    
    for (int n=0; n<5; n++)
    {
        int val;
        iss >> val>>_;
        std::cout << val*2 << '\n';
    }

     /*
    vector<string> allstrs;
    allstrs.push_back("01:01");
    allstrs.push_back("02:02");
    SolutionfindMinDifference solution;
    cout<<solution.findMinDifference(allstrs)<<endl;
    */
    
    
    
    // p1 离开作用域时， Foo 实例会自动销毁
    
    
    /*
    std::vector<std::string> x = split("one:two::three", ':');
    for(auto i:x)
        cout<<i<<endl;
    */
    /*
    vector<int> g;
    vector<int> s;
    int a[]={1,2,3};
    int b[]={1,1};
    g.assign(a,a+3);
    s.assign(b,b+2);
 
    SolutionfindContentChildrenc solution;
    cout<<solution.findContentChildren(g,s)<<endl;
     */

    /*
    SolutionfizzBuzz solution;
    vector<string> alls=solution.fizzBuzz(15);
    for(auto i:alls)
        cout<<i<<endl;
     */
    //SolutionfindComplement mysolution;
    //cout<<mysolution.findComplement(5)<<endl;
    /*
    std::priority_queue<int> q;
    vector<int> all;
    all.push_back(1);
    all.push_back(8);
    all.push_back(5);
    
    for(int n: all)
        q.push(n);
    
    print_queue(q);
    
    std::priority_queue<int, std::vector<int>, std::greater<int> > q2;
    
    for(int n:all)
        q2.push(n);
    
    print_queue(q2);
    
    // Using lambda to compare elements.
    auto cmp = [](int left, int right) { return (left ^ 1) < (right ^ 1);};
    std::priority_queue<int, std::vector<int>, decltype(cmp)> q3(cmp);
    
    for(int n : {1,8,5,6,3,4,0,9,7,2})
        q3.push(n);
    
    print_queue(q3);
     */
    /*
    vector<int> allnums;
    allnums.push_back(1);
    allnums.push_back(2);
    sort(allnums.begin(),allnums.end());
    for(auto a: allnums)
        cout<<a<<endl;
     */
    
    //SolutiontoHex solution;
    //cout<<solution.toHex(-1)<<endl;
    /*
     SolutionreadBinaryWatch solution;
     vector<string> allstrs=solution.readBinaryWatch(0);
     
     for(auto i:allstrs)
     cout<<i<<endl;
     */
    //SolutioncanConstruct solution;
    //cout<<solution.canConstruct("a", "b")<<endl;
    /*
     SolutionmaxSumSubmatrix solution;
     vector<vector<int> > testmatrix;
     int k=2;
     int matrix1[]={1,0,1};
     int matrix2[]={0,-2,3};
     testmatrix.resize(2);
     testmatrix[0].assign(matrix1,matrix1+2);
     testmatrix[1].assign(matrix2,matrix2+2);
     
     cout<<solution.maxSumSubmatrix(testmatrix, k)<<endl;
     */
    /*
     SolutionremoveInvalidParentheses mysolution;
     vector<string> allstrs=mysolution.removeInvalidParentheses("()())()");
     for(auto str: allstrs)
     cout<<str<<endl;
     */
}

#include<stack>
class SolutionnextGreaterElement {
public:
    vector<int> nextGreaterElement(vector<int>& findNums, vector<int>& nums) {
        std::stack<int> s;
        unordered_map<int, int> m;
        for (int n : nums) {
            while (s.size() && s.top() < n) {
                m[s.top()] = n;
                s.pop();
            }
            s.push(n);
        }
        vector<int> ans;
        for (int n : findNums) ans.push_back(m.count(n) ? m[n] : -1);
        return ans;
    }
};

class SolutionislandPerimeter {
public:
    int islandPerimeter(vector<vector<int>>& grid) {
        int m=(int)grid.size();
        int n=0;
        int sum=0;
        if(m>0)
            n=(int)grid[0].size();
        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++)
        {
            sum=sum+search(grid,i,j,m,n);
        }
        return sum;
    }
    
    int search(vector<vector<int> >& grid, int i, int j, const int& m, const int& n)
    {
        if(grid[i][j]==0)
            return 0;
        else
        {
            int current=4;
            if(i>=1&&grid[i-1][j]==1)
                current-=1;
            if(j>=1&&grid[i][j-1]==1)
                current-=1;
            if(i<m-1&&grid[i+1][j]==1)
                current-=1;
            if(j<n-1&&grid[i][j+1]==1)
                current-=1;
            return current;
        }
    }
};


class SolutionfindLeftMostNode {
public:
    int findLeftMostNode(TreeNode* root) {
        queue<TreeNode*> q;
        queue<int> level;
        
        q.push(root);
        level.push(0);
        
        int m=0;
        while(q.size()){
            TreeNode *r = q.front(); q.pop();
            int l = level.front(); level.pop();
            if(r->left) {
                q.push(r->left);
                level.push(l+1);
            }
            
            if(r->right){
                q.push(r->right);
                level.push(l+1);
            }
            
            if(l > m){
                m = l;
                root = r;
            }
        }
        
        return root->val;
        
    }
};

/*
 int sumOfLeftLeaves(TreeNode* root) {
 
 if(root==NULL)
 return 0;
 return iterateleft(root);
 
 }
 int iterateleft(TreeNode* root)
 {
 if(root==NULL)
 return 0;
 
 return iterateleft2(root->left)+iterateleft(root->right);
 }
 int iterateleft2(TreeNode* root)
 {
 if(root==NULL)
 return 0;
 
 if(root->left==NULL&&root->right==NULL)
 {
 return root->val;
 }
 else
 {
 return iterateleft2(root->left)+iterateleft(root->right);
 }
 }
 */


    
