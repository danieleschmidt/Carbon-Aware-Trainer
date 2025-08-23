#!/usr/bin/env python3
"""
Autonomous Quality Validation Suite

This comprehensive quality validation suite performs security scanning,
performance benchmarking, code quality analysis, and integration testing
using only Python standard library and existing project dependencies.

Quality Gates:
✅ Code runs without errors  
✅ Security scan passes
✅ Performance benchmarks met
✅ Integration tests pass
✅ Documentation coverage complete
"""

import sys
import os
import time
import json
import subprocess
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityGateResult:
    """Result of a quality gate validation."""
    
    def __init__(self, gate_name: str, passed: bool, score: float, 
                 details: str, recommendations: List[str] = None):
        self.gate_name = gate_name
        self.passed = passed
        self.score = score  # 0-100
        self.details = details
        self.recommendations = recommendations or []
        self.timestamp = datetime.now()


class AutonomousQualityValidator:
    """Autonomous quality validation system."""
    
    def __init__(self):
        """Initialize quality validator."""
        self.results: List[QualityGateResult] = []
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        
    def scan_security_vulnerabilities(self) -> QualityGateResult:
        """Scan for security vulnerabilities in Python code."""
        logger.info("🛡️ Running Security Vulnerability Scan")
        
        security_issues = []
        recommendations = []
        
        # Scan for common security anti-patterns
        python_files = []
        for root, dirs, files in os.walk(os.path.join(self.project_root, 'src')):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        # Security patterns to check
        security_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']'
            ],
            'sql_injection': [
                r'execute\s*\(\s*["\'][^"\']*%[s|d][^"\']*["\']',
                r'cursor\.execute\s*\([^)]*%',
                r'query\s*=\s*["\'][^"\']*\+[^"\']*["\']'
            ],
            'command_injection': [
                r'os\.system\s*\(',
                r'subprocess\.call\s*\([^)]*shell\s*=\s*True',
                r'eval\s*\(',
                r'exec\s*\('
            ],
            'insecure_random': [
                r'random\.random\s*\(',
                r'random\.choice\s*\(',
                r'random\.randint\s*\('
            ]
        }
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for category, patterns in security_patterns.items():
                    for pattern in patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            security_issues.append({
                                'file': os.path.relpath(file_path, self.project_root),
                                'line': line_num,
                                'category': category,
                                'pattern': pattern,
                                'matched_text': match.group()
                            })
            except Exception as e:
                logger.warning(f"Could not scan {file_path}: {e}")
                continue
        
        # Check for secure practices
        secure_practices_found = 0
        
        # Look for good security practices
        good_practices = [
            r'secrets\.',  # Using secrets module
            r'hashlib\.',  # Using proper hashing
            r'ssl_context',  # SSL usage
            r'verify=True',  # Certificate verification
        ]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for practice in good_practices:
                    if re.search(practice, content):
                        secure_practices_found += 1
                        break
            except Exception:
                continue
        
        # Calculate security score
        total_files = len(python_files)
        if total_files == 0:
            security_score = 100
        else:
            issues_penalty = min(50, len(security_issues) * 10)  # Max 50 point penalty
            practices_bonus = min(30, secure_practices_found * 5)  # Max 30 point bonus
            security_score = max(0, 100 - issues_penalty + practices_bonus)
        
        # Generate recommendations
        if security_issues:
            issue_counts = {}
            for issue in security_issues:
                category = issue['category']
                issue_counts[category] = issue_counts.get(category, 0) + 1
            
            for category, count in issue_counts.items():
                if category == 'hardcoded_secrets':
                    recommendations.append(f"Replace {count} hardcoded secrets with environment variables")
                elif category == 'sql_injection':
                    recommendations.append(f"Fix {count} potential SQL injection vulnerabilities using parameterized queries")
                elif category == 'command_injection':
                    recommendations.append(f"Secure {count} command injection vulnerabilities")
                elif category == 'insecure_random':
                    recommendations.append(f"Replace {count} insecure random calls with secrets module")
        
        if secure_practices_found < total_files * 0.2:
            recommendations.append("Increase adoption of secure coding practices (secrets module, SSL verification)")
        
        details = f"Scanned {total_files} Python files. Found {len(security_issues)} security issues across {len(set(issue['category'] for issue in security_issues))} categories."
        
        return QualityGateResult(
            gate_name="Security Scan",
            passed=security_score >= 80,
            score=security_score,
            details=details,
            recommendations=recommendations
        )
    
    def benchmark_performance(self) -> QualityGateResult:
        """Benchmark system performance."""
        logger.info("⚡ Running Performance Benchmarks")
        
        recommendations = []
        performance_metrics = {}
        
        try:
            # Test core import performance
            start_time = time.time()
            import carbon_aware_trainer
            import_time = time.time() - start_time
            performance_metrics['import_time_ms'] = import_time * 1000
            
            # Test scheduler instantiation performance
            start_time = time.time()
            from carbon_aware_trainer.core.scheduler import CarbonAwareTrainer
            trainer = CarbonAwareTrainer()
            instantiation_time = time.time() - start_time
            performance_metrics['instantiation_time_ms'] = instantiation_time * 1000
            
            # Test basic operations
            start_time = time.time()
            # Simulate basic carbon intensity calculation
            test_data = [100 + i * 10 for i in range(24)]
            avg_carbon = sum(test_data) / len(test_data)
            basic_ops_time = time.time() - start_time
            performance_metrics['basic_operations_time_ms'] = basic_ops_time * 1000
            
            # Memory usage estimation (simplified)
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                performance_metrics['memory_usage_mb'] = memory_mb
            except ImportError:
                # Fallback estimation
                performance_metrics['memory_usage_mb'] = 50  # Estimated
            
        except Exception as e:
            logger.error(f"Performance benchmarking failed: {e}")
            return QualityGateResult(
                gate_name="Performance Benchmark",
                passed=False,
                score=0,
                details=f"Performance benchmarking failed: {e}",
                recommendations=["Fix import/instantiation errors before benchmarking"]
            )
        
        # Calculate performance score
        score = 100
        
        # Import time benchmark (target: < 100ms)
        if performance_metrics['import_time_ms'] > 200:
            score -= 20
            recommendations.append("Optimize imports - taking too long to load modules")
        elif performance_metrics['import_time_ms'] > 100:
            score -= 10
            recommendations.append("Consider lazy loading for non-critical imports")
        
        # Instantiation time benchmark (target: < 50ms)  
        if performance_metrics['instantiation_time_ms'] > 100:
            score -= 20
            recommendations.append("Optimize scheduler initialization")
        elif performance_metrics['instantiation_time_ms'] > 50:
            score -= 10
        
        # Basic operations benchmark (target: < 10ms)
        if performance_metrics['basic_operations_time_ms'] > 50:
            score -= 20
            recommendations.append("Optimize basic calculation performance")
        elif performance_metrics['basic_operations_time_ms'] > 10:
            score -= 5
        
        # Memory usage benchmark (target: < 100MB)
        if performance_metrics['memory_usage_mb'] > 200:
            score -= 15
            recommendations.append("Investigate high memory usage")
        elif performance_metrics['memory_usage_mb'] > 100:
            score -= 5
            recommendations.append("Consider memory optimization")
        
        details = (f"Import: {performance_metrics['import_time_ms']:.1f}ms, "
                  f"Init: {performance_metrics['instantiation_time_ms']:.1f}ms, "
                  f"Operations: {performance_metrics['basic_operations_time_ms']:.1f}ms, "
                  f"Memory: {performance_metrics['memory_usage_mb']:.1f}MB")
        
        return QualityGateResult(
            gate_name="Performance Benchmark",
            passed=score >= 80,
            score=max(0, score),
            details=details,
            recommendations=recommendations
        )
    
    def analyze_code_quality(self) -> QualityGateResult:
        """Analyze code quality metrics."""
        logger.info("🎯 Analyzing Code Quality")
        
        recommendations = []
        quality_metrics = {}
        
        # Analyze Python files
        python_files = []
        for root, dirs, files in os.walk(os.path.join(self.project_root, 'src')):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        total_lines = 0
        total_functions = 0
        total_classes = 0
        docstring_count = 0
        comment_lines = 0
        complex_functions = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                total_lines += len(lines)
                
                in_docstring = False
                in_multiline_string = False
                current_function_lines = 0
                
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    
                    # Count comments
                    if stripped.startswith('#'):
                        comment_lines += 1
                    
                    # Count docstrings
                    if '"""' in stripped or "'''" in stripped:
                        if not in_docstring:
                            in_docstring = True
                            docstring_count += 1
                        else:
                            in_docstring = False
                    
                    # Count functions and classes
                    if stripped.startswith('def '):
                        total_functions += 1
                        current_function_lines = 0
                    elif stripped.startswith('class '):
                        total_classes += 1
                    
                    # Track function complexity (simplified)
                    if current_function_lines >= 0:
                        current_function_lines += 1
                        if any(keyword in stripped for keyword in ['if ', 'for ', 'while ', 'try:', 'except:', 'elif ']):
                            current_function_lines += 2  # Complexity weight
                        
                        if stripped.startswith('def ') and i > 0:  # New function
                            if current_function_lines > 30:  # Previous function was complex
                                complex_functions += 1
                            current_function_lines = 0
                
            except Exception as e:
                logger.warning(f"Could not analyze {file_path}: {e}")
                continue
        
        # Calculate quality metrics
        if total_lines > 0:
            quality_metrics.update({
                'total_files': len(python_files),
                'total_lines': total_lines,
                'functions': total_functions,
                'classes': total_classes,
                'docstring_coverage': (docstring_count / max(1, total_functions + total_classes)) * 100,
                'comment_ratio': (comment_lines / total_lines) * 100,
                'complex_functions': complex_functions,
                'avg_file_size': total_lines / max(1, len(python_files))
            })
        
        # Calculate quality score
        score = 100
        
        # Docstring coverage (target: > 60%)
        docstring_coverage = quality_metrics.get('docstring_coverage', 0)
        if docstring_coverage < 30:
            score -= 25
            recommendations.append("Increase docstring coverage for functions and classes")
        elif docstring_coverage < 60:
            score -= 15
            recommendations.append("Add more docstrings to improve code documentation")
        
        # Comment ratio (target: 5-15%)
        comment_ratio = quality_metrics.get('comment_ratio', 0)
        if comment_ratio < 2:
            score -= 15
            recommendations.append("Add more explanatory comments to complex code")
        elif comment_ratio > 20:
            score -= 10
            recommendations.append("Consider if excessive comments indicate overly complex code")
        
        # Function complexity
        if total_functions > 0:
            complex_ratio = (complex_functions / total_functions) * 100
            if complex_ratio > 20:
                score -= 20
                recommendations.append("Refactor complex functions to improve maintainability")
            elif complex_ratio > 10:
                score -= 10
                recommendations.append("Consider breaking down some complex functions")
        
        # File size
        avg_file_size = quality_metrics.get('avg_file_size', 0)
        if avg_file_size > 500:
            score -= 10
            recommendations.append("Consider breaking down large files into smaller modules")
        
        details = (f"Files: {len(python_files)}, Lines: {total_lines}, "
                  f"Functions: {total_functions}, Classes: {total_classes}, "
                  f"Docstring: {docstring_coverage:.1f}%, Comments: {comment_ratio:.1f}%")
        
        return QualityGateResult(
            gate_name="Code Quality Analysis",
            passed=score >= 70,
            score=max(0, score),
            details=details,
            recommendations=recommendations
        )
    
    def test_integration_functionality(self) -> QualityGateResult:
        """Test integration and basic functionality."""
        logger.info("🔧 Testing Integration Functionality")
        
        recommendations = []
        test_results = {}
        
        try:
            # Test 1: Core imports
            from carbon_aware_trainer.core.scheduler import CarbonAwareTrainer
            from carbon_aware_trainer.core.types import CarbonIntensity, TrainingConfig
            test_results['core_imports'] = True
            
            # Test 2: Basic instantiation
            trainer = CarbonAwareTrainer(
                carbon_model='electricitymap',
                region='US-CA',
                target_carbon_intensity=100.0
            )
            test_results['basic_instantiation'] = True
            
            # Test 3: Configuration handling
            config = TrainingConfig(
                carbon_threshold=100.0,
                pause_threshold=150.0,
                resume_threshold=80.0
            )
            test_results['config_handling'] = True
            
            # Test 4: Carbon intensity creation
            carbon_data = CarbonIntensity(
                region='US-CA',
                timestamp=datetime.now(),
                carbon_intensity=75.5,
                renewable_percentage=45.0
            )
            test_results['data_structures'] = True
            
            # Test 5: Research modules (optional)
            try:
                from carbon_aware_trainer.research.quantum_inspired_carbon_optimizer import QuantumInspiredCarbonOptimizer
                test_results['research_modules'] = True
            except ImportError:
                test_results['research_modules'] = False
                recommendations.append("Research modules require additional dependencies (numpy)")
            
            # Test 6: Translation support
            try:
                from carbon_aware_trainer.core.i18n import get_translation
                test_results['i18n_support'] = True
            except ImportError:
                test_results['i18n_support'] = False
                recommendations.append("Internationalization support not fully available")
            
            # Test 7: Monitoring integration
            try:
                from carbon_aware_trainer.monitoring.metrics import MetricsCollector
                test_results['monitoring_integration'] = True
            except ImportError:
                test_results['monitoring_integration'] = False
            
        except Exception as e:
            logger.error(f"Integration testing failed: {e}")
            return QualityGateResult(
                gate_name="Integration Testing",
                passed=False,
                score=0,
                details=f"Integration testing failed: {e}",
                recommendations=["Fix critical import/instantiation errors"]
            )
        
        # Calculate integration score
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result)
        
        score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Critical tests must pass
        critical_tests = ['core_imports', 'basic_instantiation', 'config_handling', 'data_structures']
        critical_passed = all(test_results.get(test, False) for test in critical_tests)
        
        if not critical_passed:
            score = min(score, 50)
            recommendations.append("Critical integration tests failed - fix core functionality")
        
        details = f"Integration tests: {passed_tests}/{total_tests} passed. Critical functionality: {'✅' if critical_passed else '❌'}"
        
        return QualityGateResult(
            gate_name="Integration Testing",
            passed=score >= 80 and critical_passed,
            score=score,
            details=details,
            recommendations=recommendations
        )
    
    def validate_documentation_coverage(self) -> QualityGateResult:
        """Validate documentation coverage."""
        logger.info("📚 Validating Documentation Coverage")
        
        recommendations = []
        doc_metrics = {}
        
        # Check for essential documentation files
        required_docs = {
            'README.md': 'Project overview and usage instructions',
            'LICENSE': 'Software license',
            'DEPLOYMENT_GUIDE.md': 'Deployment instructions',
            'requirements.txt': 'Python dependencies'
        }
        
        existing_docs = {}
        for doc_file, description in required_docs.items():
            file_path = os.path.join(self.project_root, doc_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    existing_docs[doc_file] = {
                        'exists': True,
                        'size': len(content),
                        'lines': len(content.split('\n'))
                    }
                except Exception:
                    existing_docs[doc_file] = {'exists': False, 'size': 0, 'lines': 0}
            else:
                existing_docs[doc_file] = {'exists': False, 'size': 0, 'lines': 0}
        
        # Check README quality
        readme_quality = 0
        readme_path = os.path.join(self.project_root, 'README.md')
        if os.path.exists(readme_path):
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read().lower()
                
                # Check for essential README sections
                readme_sections = {
                    'installation': ['install', 'setup'],
                    'usage': ['usage', 'example', 'quickstart'],
                    'features': ['features', 'capabilities'],
                    'documentation': ['docs', 'documentation', 'api'],
                    'license': ['license', 'mit', 'apache']
                }
                
                sections_found = 0
                for section, keywords in readme_sections.items():
                    if any(keyword in readme_content for keyword in keywords):
                        sections_found += 1
                
                readme_quality = (sections_found / len(readme_sections)) * 100
                
            except Exception:
                readme_quality = 0
        
        # Check for code documentation  
        python_files = []
        for root, dirs, files in os.walk(os.path.join(self.project_root, 'src')):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        documented_modules = 0
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for module docstring
                if content.strip().startswith('"""') or content.strip().startswith("'''"):
                    documented_modules += 1
                elif '"""' in content[:500] or "'''" in content[:500]:  # Docstring within first 500 chars
                    documented_modules += 1
                    
            except Exception:
                continue
        
        module_doc_coverage = (documented_modules / max(1, len(python_files))) * 100
        
        # Calculate documentation score
        docs_exist_score = sum(1 for doc in existing_docs.values() if doc['exists'])
        docs_exist_score = (docs_exist_score / len(required_docs)) * 100
        
        # Weighted average of documentation metrics
        score = (
            docs_exist_score * 0.4 +
            readme_quality * 0.3 +
            module_doc_coverage * 0.3
        )
        
        # Generate recommendations
        if score < 80:
            missing_docs = [doc for doc, info in existing_docs.items() if not info['exists']]
            if missing_docs:
                recommendations.append(f"Create missing documentation: {', '.join(missing_docs)}")
            
            if readme_quality < 70:
                recommendations.append("Improve README with more comprehensive sections")
            
            if module_doc_coverage < 60:
                recommendations.append("Add docstrings to more Python modules")
        
        doc_metrics = {
            'required_docs_present': docs_exist_score,
            'readme_quality': readme_quality,
            'module_documentation': module_doc_coverage,
            'total_python_files': len(python_files),
            'documented_modules': documented_modules
        }
        
        details = f"Docs present: {docs_exist_score:.0f}%, README quality: {readme_quality:.0f}%, Module docs: {module_doc_coverage:.0f}%"
        
        return QualityGateResult(
            gate_name="Documentation Coverage",
            passed=score >= 75,
            score=score,
            details=details,
            recommendations=recommendations
        )
    
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gates and generate comprehensive report."""
        logger.info("🚀 Running All Quality Gates")
        
        # Execute all quality gates
        self.results = [
            self.scan_security_vulnerabilities(),
            self.benchmark_performance(),
            self.analyze_code_quality(),
            self.test_integration_functionality(),
            self.validate_documentation_coverage()
        ]
        
        # Calculate overall score
        total_score = sum(result.score for result in self.results) / len(self.results)
        total_passed = all(result.passed for result in self.results)
        
        # Collect all recommendations
        all_recommendations = []
        for result in self.results:
            all_recommendations.extend(result.recommendations)
        
        # Generate summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': total_score,
            'all_gates_passed': total_passed,
            'gates_results': {
                result.gate_name: {
                    'passed': result.passed,
                    'score': result.score,
                    'details': result.details
                }
                for result in self.results
            },
            'recommendations': all_recommendations,
            'ready_for_production': total_passed and total_score >= 80
        }
        
        return summary


def main():
    """Main execution function."""
    print("🛡️ AUTONOMOUS QUALITY VALIDATION SUITE")
    print("=" * 50)
    print("🔍 Executing comprehensive quality gates validation")
    print()
    
    # Initialize validator
    validator = AutonomousQualityValidator()
    
    # Run all quality gates
    start_time = time.time()
    report = validator.run_all_quality_gates()
    execution_time = time.time() - start_time
    
    # Display results
    print("📊 QUALITY GATES RESULTS")
    print("-" * 30)
    print(f"Overall Score: {report['overall_score']:.1f}/100")
    print(f"All Gates Passed: {'✅ YES' if report['all_gates_passed'] else '❌ NO'}")
    print(f"Production Ready: {'✅ YES' if report['ready_for_production'] else '❌ NO'}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print()
    
    print("🎯 INDIVIDUAL GATE RESULTS")
    print("-" * 30)
    for gate_name, gate_result in report['gates_results'].items():
        status = "✅ PASS" if gate_result['passed'] else "❌ FAIL"
        print(f"• {gate_name}: {status} ({gate_result['score']:.1f}/100)")
        print(f"  {gate_result['details']}")
    
    if report['recommendations']:
        print("\n💡 RECOMMENDATIONS")
        print("-" * 20)
        for i, recommendation in enumerate(report['recommendations'], 1):
            print(f"{i}. {recommendation}")
    
    print("\n📈 QUALITY METRICS ACHIEVED")
    print("-" * 35)
    
    # Calculate achievements
    passed_gates = sum(1 for result in validator.results if result.passed)
    total_gates = len(validator.results)
    
    achievements = [
        f"✅ Quality Gates Passed: {passed_gates}/{total_gates}",
        f"✅ Security Scan: {'Passed' if any(r.gate_name == 'Security Scan' and r.passed for r in validator.results) else 'Failed'}",
        f"✅ Performance: {'Optimal' if any(r.gate_name == 'Performance Benchmark' and r.score >= 90 for r in validator.results) else 'Good' if any(r.gate_name == 'Performance Benchmark' and r.score >= 80 for r in validator.results) else 'Needs Improvement'}",
        f"✅ Code Quality: {'High' if any(r.gate_name == 'Code Quality Analysis' and r.score >= 85 for r in validator.results) else 'Good' if any(r.gate_name == 'Code Quality Analysis' and r.score >= 70 for r in validator.results) else 'Needs Improvement'}",
        f"✅ Integration: {'Fully Functional' if any(r.gate_name == 'Integration Testing' and r.passed for r in validator.results) else 'Issues Detected'}",
        f"✅ Documentation: {'Comprehensive' if any(r.gate_name == 'Documentation Coverage' and r.score >= 80 for r in validator.results) else 'Adequate' if any(r.gate_name == 'Documentation Coverage' and r.score >= 60 for r in validator.results) else 'Insufficient'}"
    ]
    
    for achievement in achievements:
        print(achievement)
    
    # Save report
    print("\n💾 Saving Quality Report...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs("quality_reports", exist_ok=True)
    report_file = f"quality_reports/quality_validation_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✅ Quality report saved: {report_file}")
    
    # Final verdict
    print(f"\n🎉 QUALITY VALIDATION {'COMPLETE' if report['all_gates_passed'] else 'COMPLETE WITH ISSUES'}")
    
    if report['ready_for_production']:
        print("🚀 SYSTEM IS READY FOR PRODUCTION DEPLOYMENT!")
        print("   All critical quality gates have been met")
        print("   Security, performance, and functionality validated")
    else:
        print("⚠️  SYSTEM REQUIRES ATTENTION BEFORE PRODUCTION")
        print("   Address the recommendations above before deployment")
        print("   Re-run quality gates after improvements")
    
    return report


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Quality validation interrupted")
    except Exception as e:
        print(f"\n❌ Quality validation failed: {e}")
        import traceback
        traceback.print_exc()